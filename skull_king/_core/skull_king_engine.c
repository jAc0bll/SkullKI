/*
 * skull_king_engine.c — Full C game engine + MLP inference + CFR traversal.
 *
 * Exports (Python module: skull_king_engine):
 *   set_adv_weights(w1, b1, w2, b2, w3, b3)  — load float32 numpy weight arrays
 *   traverse(traverser, seed, n_players, heuristic_frac) → 7-tuple of numpy arrays
 *
 * Build (from skull_king/_core/):
 *   Windows MSVC : set DISTUTILS_USE_SDK=1 && set MSSdk=1 && python setup_engine.py build_ext --inplace
 *   Linux gcc    : python setup_engine.py build_ext --inplace
 *
 * Integer encodings (must match skull_king/trick.py and skull_king/env/skull_king_env.py):
 *   Card type : NUMBERED=0 ESCAPE=1 PIRATE=2 MERMAID=3 SKULL_KING=4 TIGRESS=5
 *   Suit      : BLACK=0 YELLOW=1 GREEN=2 PURPLE=3 NONE=-1
 *   Slots     : 0-55 numbered(suit*14+val-1), 56-60 Escape, 61-65 Pirate,
 *               66-67 Mermaid, 68 SkullKing, 69 Tigress
 *   Actions   : 0-10 bid(=amount), 11+slot play, 80 Tigress/Escape, 81 Tigress/Pirate
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>

/* ─── AVX2 ──────────────────────────────────────────────────────────────── */
#if defined(__AVX2__)
#  include <immintrin.h>
#  define USE_AVX2 1
#else
#  define USE_AVX2 0
#endif

/* ─── Game constants ─────────────────────────────────────────────────────── */
#define N_PLAYERS_MAX   6
#define N_ROUNDS        10
#define DECK_SIZE       70
#define ACTION_SIZE     82
#define OBS_SIZE        244
#define N_BID_ACTIONS   11
#define ACT_TIG_ESCAPE  80
#define ACT_TIG_PIRATE  81

/* Card types */
#define CT_NUMBERED   0
#define CT_ESCAPE     1
#define CT_PIRATE     2
#define CT_MERMAID    3
#define CT_SKULL_KING 4
#define CT_TIGRESS    5

/* Suits */
#define SUIT_BLACK    0
#define SUIT_NONE    -1

/* Phases */
#define PHASE_BIDDING   0
#define PHASE_PLAYING   1
#define PHASE_GAMEOVER  2

/* MLP dimensions (w1 padded to multiple of 8 for AVX2) */
#define MLP_IN_RAW  244   /* actual observation size */
#define MLP_IN      248   /* padded input (244 → 248 = 31*8) */
#define MLP_H1      512
#define MLP_H2      512
#define MLP_OUT      82

/* Split-net dimensions */
#define BID_H1    256
#define BID_H2    256
#define BID_OUT    11   /* bid actions 0-10 */
#define PLAY_H1   512
#define PLAY_H2   512
#define PLAY_OUT   71   /* play: slots 0-68 + TIG_ESCAPE(69) + TIG_PIRATE(70) */

/* ─── Slot → card property lookups (inlined) ─────────────────────────────── */
static inline int slot_ctype(int s)  { return (s<56)?CT_NUMBERED:(s<61)?CT_ESCAPE:(s<66)?CT_PIRATE:(s<68)?CT_MERMAID:(s==68)?CT_SKULL_KING:CT_TIGRESS; }
static inline int slot_suit(int s)   { return (s<56)?(s/14):SUIT_NONE; }
static inline int slot_value(int s)  { return (s<56)?(s%14+1):0; }
static inline int eff_type(int s, int tig) { int t=slot_ctype(s); return (t==CT_TIGRESS)?((tig==1)?CT_PIRATE:CT_ESCAPE):t; }

/* ─── GameState ──────────────────────────────────────────────────────────── */
typedef struct {
    int8_t   n_players;
    int8_t   round;            /* 1-10 */
    int8_t   trick_in_round;   /* 1-based */
    int8_t   trick_leader;     /* player index who leads current trick */
    int8_t   play_count;       /* cards played so far in current trick */
    int8_t   phase;            /* PHASE_* */
    uint8_t  bids_placed;      /* bitmask: bit i = player i has bid */
    int8_t   _pad;

    uint8_t  hand[N_PLAYERS_MAX][N_ROUNDS]; /* canonical slot, 0xFF=empty */
    int8_t   hand_size[N_PLAYERS_MAX];

    int8_t   bid[N_PLAYERS_MAX];         /* -1 = not yet bid */
    int8_t   tricks_won[N_PLAYERS_MAX];
    int16_t  round_bonus[N_PLAYERS_MAX];
    int16_t  total_score[N_PLAYERS_MAX];

    /* Current trick (up to N_PLAYERS_MAX cards) */
    uint8_t  trick_slot[N_PLAYERS_MAX];
    uint8_t  trick_tig[N_PLAYERS_MAX];    /* 0=N/A, 1=pirate, 2=escape */
    int8_t   trick_player[N_PLAYERS_MAX];
    int8_t   trick_order[N_PLAYERS_MAX];  /* 1-based play order */

    /* Seen cards from completed tricks this round (canonical slot bitmap) */
    uint8_t  seen[DECK_SIZE];

    uint64_t rng;  /* xorshift64 state */
} GameState;

/* ─── MLP weights ────────────────────────────────────────────────────────── */
typedef struct {
    float w1[MLP_H1][MLP_IN];
    float b1[MLP_H1];
    float w2[MLP_H2][MLP_H1];
    float b2[MLP_H2];
    float w3[MLP_OUT][MLP_H2];
    float b3[MLP_OUT];
} MLPWeights;

typedef struct {
    float w1[BID_H1][MLP_IN];
    float b1[BID_H1];
    float w2[BID_H2][BID_H1];
    float b2[BID_H2];
    float w3[BID_OUT][BID_H2];
    float b3[BID_OUT];
} BidMLPWeights;

typedef struct {
    float w1[PLAY_H1][MLP_IN];
    float b1[PLAY_H1];
    float w2[PLAY_H2][PLAY_H1];
    float b2[PLAY_H2];
    float w3[PLAY_OUT][PLAY_H2];
    float b3[PLAY_OUT];
} PlayMLPWeights;

/* Global advantage-net weights (loaded once per worker via set_adv_weights) */
static MLPWeights g_adv;
static int        g_weights_loaded = 0;
static BidMLPWeights  g_bid_adv;
static PlayMLPWeights g_play_adv;
static int            g_bid_loaded  = 0;
static int            g_play_loaded = 0;

/* ─── RNG (xorshift64) ───────────────────────────────────────────────────── */
static inline uint64_t rng_next(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return (*s = x);
}
/* Uniform float in [0, 1) */
static inline float rng_float(uint64_t *s) {
    return (float)(rng_next(s) >> 11) * (1.0f / (float)(1ULL << 53));
}
/* Uniform integer [0, n) */
static inline int rng_int(uint64_t *s, int n) {
    return (int)(rng_next(s) % (uint64_t)n);
}

/* ─── GameState helpers ──────────────────────────────────────────────────── */

static void deal_round(GameState *gs) {
    int n = gs->n_players, r = gs->round;
    for (int p = 0; p < n; p++) {
        gs->bid[p] = -1;
        gs->tricks_won[p] = 0;
        gs->round_bonus[p] = 0;
        gs->hand_size[p] = (int8_t)r;
        for (int c = 0; c < N_ROUNDS; c++) gs->hand[p][c] = 0xFF;
    }
    memset(gs->seen, 0, DECK_SIZE);
    gs->bids_placed    = 0;
    gs->trick_in_round = 1;
    gs->play_count     = 0;
    for (int i = 0; i < N_PLAYERS_MAX; i++) {
        gs->trick_slot[i]   = 0xFF;
        gs->trick_tig[i]    = 0;
        gs->trick_player[i] = -1;
        gs->trick_order[i]  = 0;
    }

    /* Fisher-Yates shuffle of canonical deck */
    uint8_t deck[DECK_SIZE];
    for (int i = 0; i < DECK_SIZE; i++) deck[i] = (uint8_t)i;
    for (int i = DECK_SIZE - 1; i > 0; i--) {
        int j = rng_int(&gs->rng, i + 1);
        uint8_t t = deck[i]; deck[i] = deck[j]; deck[j] = t;
    }
    for (int p = 0; p < n; p++)
        for (int c = 0; c < r; c++)
            gs->hand[p][c] = deck[p * r + c];
}

static void gs_init(GameState *gs, int n_players, uint64_t seed) {
    memset(gs, 0, sizeof(GameState));
    gs->n_players    = (int8_t)n_players;
    gs->round        = 1;
    gs->trick_leader = 0;
    gs->phase        = PHASE_BIDDING;
    gs->rng          = seed ? seed : 12345678901ULL;
    for (int p = 0; p < N_PLAYERS_MAX; p++) gs->bid[p] = -1;
    deal_round(gs);
}

static int gs_current_player(const GameState *gs) {
    if (gs->phase == PHASE_BIDDING) {
        for (int i = 0; i < gs->n_players; i++)
            if (!(gs->bids_placed & (uint8_t)(1u << i))) return i;
        return 0;
    }
    if (gs->phase == PHASE_PLAYING)
        return (gs->trick_leader + gs->play_count) % gs->n_players;
    return 0;
}

static void gs_legal_mask(const GameState *gs, uint8_t mask[ACTION_SIZE]) {
    memset(mask, 0, ACTION_SIZE);
    int p = gs_current_player(gs);

    if (gs->phase == PHASE_BIDDING) {
        for (int b = 0; b <= gs->round; b++) mask[b] = 1;
        return;
    }

    /* Find led suit: suit of first NUMBERED card by play_order */
    int led_suit = SUIT_NONE, first_order = 255;
    for (int i = 0; i < gs->play_count; i++) {
        int s = gs->trick_slot[i];
        if (s == 0xFF) continue;
        if (eff_type(s, gs->trick_tig[i]) == CT_NUMBERED
                && gs->trick_order[i] < first_order) {
            first_order = gs->trick_order[i];
            led_suit    = slot_suit(s);
        }
    }

    /* Check void in led suit */
    int has_led = 0;
    if (led_suit != SUIT_NONE) {
        for (int c = 0; c < gs->hand_size[p]; c++) {
            int s = gs->hand[p][c];
            if (s != 0xFF && slot_ctype(s) == CT_NUMBERED && slot_suit(s) == led_suit) {
                has_led = 1; break;
            }
        }
    }

    for (int c = 0; c < gs->hand_size[p]; c++) {
        int s = gs->hand[p][c];
        if (s == 0xFF) continue;
        int t = slot_ctype(s);

        if (t == CT_TIGRESS) {
            mask[ACT_TIG_ESCAPE] = 1;
            mask[ACT_TIG_PIRATE] = 1;
            continue;
        }
        /* Suit-following constraint */
        if (led_suit != SUIT_NONE && has_led && t == CT_NUMBERED && slot_suit(s) != led_suit)
            continue;
        mask[N_BID_ACTIONS + s] = 1;
    }
}

/* ─── Trick resolution ───────────────────────────────────────────────────── */

/* Resolve winner index into trick arrays; raw arrays (to support would_beat) */
static int resolve_winner_raw(
    const uint8_t *slots, const uint8_t *tigs,
    const int8_t *orders, int n)
{
    int has_sk=0, has_mer=0, has_pir=0;
    for (int i = 0; i < n; i++) {
        int et = eff_type(slots[i], tigs[i]);
        if (et == CT_SKULL_KING) has_sk  = 1;
        if (et == CT_MERMAID)    has_mer = 1;
        if (et == CT_PIRATE)     has_pir = 1;
    }

/* Macro: first of given effective type by play_order */
#define FIRST_OF(TYPE, OUT)                                         \
    do {                                                            \
        int _bst = -1, _bo = 255;                                   \
        for (int _i = 0; _i < n; _i++)                             \
            if (eff_type(slots[_i],tigs[_i])==(TYPE) && orders[_i]<_bo) \
                { _bst=_i; _bo=orders[_i]; }                       \
        (OUT) = _bst;                                               \
    } while(0)

    int w;
    if (has_sk && has_mer && !has_pir) { FIRST_OF(CT_MERMAID, w); return w; }
    if (has_sk)                         { FIRST_OF(CT_SKULL_KING, w); return w; }
    if (has_pir)                        { FIRST_OF(CT_PIRATE, w); return w; }
    if (has_mer)                        { FIRST_OF(CT_MERMAID, w); return w; }
#undef FIRST_OF

    /* Numbered/Escape resolution */
    int all_esc = 1;
    for (int i = 0; i < n; i++)
        if (eff_type(slots[i], tigs[i]) != CT_ESCAPE) { all_esc = 0; break; }
    if (all_esc) {
        int best = 0;
        for (int i = 1; i < n; i++) if (orders[i] < orders[best]) best = i;
        return best;
    }

    /* Led suit */
    int led_suit = SUIT_NONE, f_ord = 255;
    for (int i = 0; i < n; i++) {
        int s = slots[i];
        if (eff_type(s, tigs[i]) == CT_NUMBERED && orders[i] < f_ord) {
            f_ord = orders[i]; led_suit = slot_suit(s);
        }
    }

    if (led_suit == SUIT_NONE) {
        /* Escape led: highest BLACK wins, else informal led suit */
        int bb = -1, bv = -1;
        for (int i = 0; i < n; i++) {
            int s = slots[i];
            if (eff_type(s, tigs[i]) == CT_NUMBERED && slot_suit(s) == SUIT_BLACK) {
                int v = slot_value(s);
                if (v > bv) { bv = v; bb = i; }
            }
        }
        if (bb >= 0) return bb;
        /* First non-escape colored card sets informal led suit */
        int fc = -1; f_ord = 255;
        for (int i = 0; i < n; i++) {
            if (eff_type(slots[i], tigs[i]) != CT_ESCAPE && orders[i] < f_ord)
                { f_ord = orders[i]; fc = i; }
        }
        led_suit = (fc >= 0) ? slot_suit(slots[fc]) : SUIT_NONE;
    }

    /* Highest BLACK trump */
    { int bb = -1, bv = -1;
      for (int i = 0; i < n; i++) {
          int s = slots[i];
          if (eff_type(s, tigs[i]) == CT_NUMBERED && slot_suit(s) == SUIT_BLACK) {
              int v = slot_value(s);
              if (v > bv) { bv = v; bb = i; }
          }
      }
      if (bb >= 0) return bb;
    }

    /* Highest card of led suit */
    { int bl = -1, bv = -1;
      for (int i = 0; i < n; i++) {
          int s = slots[i];
          if (eff_type(s, tigs[i]) == CT_NUMBERED && slot_suit(s) == led_suit) {
              int v = slot_value(s);
              if (v > bv) { bv = v; bl = i; }
          }
      }
      return bl;  /* guaranteed non-negative */
    }
}

static int compute_bonus(const uint8_t *slots, const uint8_t *tigs, int n, int wi) {
    int bonus = 0;
    int wt = eff_type(slots[wi], tigs[wi]);

    if (wt == CT_SKULL_KING || wt == CT_MERMAID || wt == CT_PIRATE) {
        for (int i = 0; i < n; i++) {
            if (eff_type(slots[i], tigs[i]) == CT_NUMBERED
                    && slot_suit(slots[i]) == SUIT_BLACK
                    && slot_value(slots[i]) == 14) { bonus += 20; break; }
        }
    }
    if (wt == CT_SKULL_KING) {
        for (int i = 0; i < n; i++)
            if (eff_type(slots[i], tigs[i]) == CT_PIRATE) bonus += 30;
    }
    if (wt == CT_MERMAID) {
        for (int i = 0; i < n; i++)
            if (eff_type(slots[i], tigs[i]) == CT_SKULL_KING) { bonus += 40; break; }
    }
    return bonus;
}

static void resolve_trick(GameState *gs) {
    int n = gs->play_count;
    int wi = resolve_winner_raw(gs->trick_slot, gs->trick_tig, gs->trick_order, n);
    int wp = gs->trick_player[wi];
    int bon = compute_bonus(gs->trick_slot, gs->trick_tig, n, wi);

    gs->tricks_won[wp]++;
    gs->round_bonus[wp] += (int16_t)bon;

    for (int i = 0; i < n; i++) {
        int s = gs->trick_slot[i];
        if (s < DECK_SIZE) gs->seen[s] = 1;
    }

    gs->trick_leader = (int8_t)wp;
    gs->trick_in_round++;
    gs->play_count = 0;
    for (int i = 0; i < N_PLAYERS_MAX; i++) {
        gs->trick_slot[i]   = 0xFF;
        gs->trick_tig[i]    = 0;
        gs->trick_player[i] = -1;
        gs->trick_order[i]  = 0;
    }

    if (gs->trick_in_round > gs->round) {
        /* End round: finalize scores */
        for (int p = 0; p < gs->n_players; p++) {
            int bid = gs->bid[p], won = gs->tricks_won[p];
            int bon2 = gs->round_bonus[p], rn = gs->round;
            int rs;
            if (bid == 0) {
                rs = (won == 0) ? rn * 10 : rn * (-10);
            } else if (won == bid) {
                rs = bid * 20 + bon2;
            } else {
                rs = abs(bid - won) * (-10);
            }
            gs->total_score[p] += (int16_t)rs;
        }

        if (gs->round == N_ROUNDS) {
            gs->phase = PHASE_GAMEOVER;
        } else {
            gs->round++;
            gs->phase = PHASE_BIDDING;
            deal_round(gs);
        }
    }
}

static void gs_apply_action(GameState *gs, int action) {
    int p = gs_current_player(gs);

    if (gs->phase == PHASE_BIDDING) {
        gs->bid[p] = (int8_t)action;
        gs->bids_placed |= (uint8_t)(1u << p);
        int cnt = 0;
        uint8_t b = gs->bids_placed;
        while (b) { cnt += b & 1; b >>= 1; }
        if (cnt == gs->n_players) gs->phase = PHASE_PLAYING;
        return;
    }

    uint8_t slot; uint8_t tig = 0;
    if (action == ACT_TIG_ESCAPE)  { slot = 69; tig = 2; }
    else if (action == ACT_TIG_PIRATE) { slot = 69; tig = 1; }
    else { slot = (uint8_t)(action - N_BID_ACTIONS); }

    /* Remove from hand */
    int hs = gs->hand_size[p];
    for (int c = 0; c < hs; c++) {
        if (gs->hand[p][c] == slot) {
            gs->hand[p][c] = gs->hand[p][hs - 1];
            gs->hand[p][hs - 1] = 0xFF;
            gs->hand_size[p]--;
            break;
        }
    }

    int tc = gs->play_count;
    gs->trick_slot[tc]   = slot;
    gs->trick_tig[tc]    = tig;
    gs->trick_player[tc] = (int8_t)p;
    gs->trick_order[tc]  = (int8_t)(tc + 1);
    gs->play_count++;

    if (gs->play_count == gs->n_players) resolve_trick(gs);
}

static float gs_utility(const GameState *gs, int player) {
    float my = (float)gs->total_score[player];
    int n = gs->n_players;
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
        if (i != player) sum += (float)gs->total_score[i];
    float avg = sum / (float)(n - 1);
    return (my - avg) / (30.0f * (float)n);
}

static void gs_build_obs(const GameState *gs, int player, float obs[MLP_IN]) {
    memset(obs, 0, MLP_IN * sizeof(float));
    int p = player, n = gs->n_players;
    float rn_f = (float)gs->round;

    /* [0:70] hand */
    for (int c = 0; c < gs->hand_size[p]; c++) {
        int s = gs->hand[p][c];
        if (s < DECK_SIZE) obs[s] = 1.0f;
    }
    /* [70:140] current trick */
    for (int i = 0; i < gs->play_count; i++) {
        int s = gs->trick_slot[i];
        if (s < DECK_SIZE) obs[70 + s] = 1.0f;
    }
    /* [140:210] seen */
    for (int s = 0; s < DECK_SIZE; s++)
        if (gs->seen[s]) obs[140 + s] = 1.0f;

    /* [210..239] player stats, relative to current player */
    for (int i = 0; i < N_PLAYERS_MAX; i++) {
        int actual = (i < n) ? (p + i) % n : -1;
        if (actual == -1) { obs[210 + i] = -1.0f; continue; }
        int8_t bid = gs->bid[actual];
        if (bid < 0) {
            obs[210 + i] = -1.0f;
        } else {
            obs[210 + i] = bid / rn_f;
            obs[228 + i] = 1.0f;
        }
        obs[216 + i] = gs->tricks_won[actual] / rn_f;
        float sc = gs->total_score[actual] / 300.0f;
        if (sc >  1.0f) sc =  1.0f;
        if (sc < -1.0f) sc = -1.0f;
        obs[222 + i] = sc;
        if (actual == gs->trick_leader) obs[234 + i] = 1.0f;
    }

    obs[240] = (gs->round - 1) / 9.0f;
    obs[241] = (gs->trick_in_round - 1) / (float)(gs->round > 1 ? gs->round - 1 : 1);
    obs[242] = (gs->phase == PHASE_BIDDING) ? 1.0f : 0.0f;
    obs[243] = gs->play_count / (float)n;
    /* obs[244..247] = 0 (AVX2 padding, already zeroed) */
}

/* ─── Heuristic opponent (simplified) ───────────────────────────────────── */

static inline int card_strength(int slot, int tig) {
    int t = slot_ctype(slot);
    if (t == CT_SKULL_KING) return 10000;
    if (t == CT_PIRATE)     return 9000;
    if (t == CT_TIGRESS)    return (tig == 1) ? 9000 : 0;
    if (t == CT_MERMAID)    return 5000;
    if (t == CT_ESCAPE)     return 0;
    return (slot_suit(slot) == SUIT_BLACK ? 1000 : 0) + slot_value(slot) * 10;
}

/* Would (slot, tig) beat the current trick if played as the next card? */
static int would_beat(const GameState *gs, int slot, int tig) {
    int n = gs->play_count;
    if (n == 0) return 1;

    /* Build extended arrays */
    int p = gs_current_player(gs);
    uint8_t t_slots[N_PLAYERS_MAX]; uint8_t t_tigs[N_PLAYERS_MAX];
    int8_t  t_orders[N_PLAYERS_MAX]; int8_t t_players[N_PLAYERS_MAX];
    for (int i = 0; i < n; i++) {
        t_slots[i]   = gs->trick_slot[i];
        t_tigs[i]    = gs->trick_tig[i];
        t_orders[i]  = gs->trick_order[i];
        t_players[i] = gs->trick_player[i];
    }
    t_slots[n]   = (uint8_t)slot;
    t_tigs[n]    = (uint8_t)tig;
    t_orders[n]  = (int8_t)(n + 1);
    t_players[n] = (int8_t)p;
    int wi = resolve_winner_raw(t_slots, t_tigs, t_orders, n + 1);
    return t_players[wi] == p;
}

static int heuristic_bid(const GameState *gs, int p) {
    float w = 0.0f;
    for (int c = 0; c < gs->hand_size[p]; c++) {
        int s = gs->hand[p][c]; if (s == 0xFF) continue;
        int t = slot_ctype(s);
        if (t == CT_SKULL_KING) w += 0.95f;
        else if (t == CT_PIRATE)  w += 0.75f;
        else if (t == CT_TIGRESS) w += 0.45f;
        else if (t == CT_MERMAID) w += 0.25f;
        else if (t == CT_ESCAPE)  w += 0.00f;
        else { /* NUMBERED */
            int v = slot_value(s), su = slot_suit(s);
            w += (su == SUIT_BLACK) ? (0.15f + (v/14.0f)*0.60f) : (v/14.0f)*0.15f;
        }
    }
    int bid = (int)(w + 0.5f);
    if (bid > gs->round) bid = gs->round;
    if (bid < 0) bid = 0;
    return bid;
}

static int heuristic_play(const GameState *gs, int p) {
    uint8_t mask[ACTION_SIZE];
    gs_legal_mask(gs, mask);
    int want_win = (gs->bid[p] > 0 && gs->tricks_won[p] < gs->bid[p]);

    /* Collect legal play actions */
    int l_slots[N_ROUNDS + 2], l_tigs[N_ROUNDS + 2], l_acts[N_ROUNDS + 2], nl = 0;
    for (int a = N_BID_ACTIONS; a < ACTION_SIZE; a++) {
        if (!mask[a]) continue;
        int sl, tg;
        if (a == ACT_TIG_ESCAPE) { sl = 69; tg = 2; }
        else if (a == ACT_TIG_PIRATE) { sl = 69; tg = 1; }
        else { sl = a - N_BID_ACTIONS; tg = 0; }
        l_slots[nl] = sl; l_tigs[nl] = tg; l_acts[nl] = a; nl++;
    }
    if (nl == 0) return N_BID_ACTIONS;
    if (nl == 1) return l_acts[0];

    if (want_win) {
        /* Weakest card that beats the trick */
        int best = -1, best_str = INT_MAX;
        for (int i = 0; i < nl; i++) {
            if (would_beat(gs, l_slots[i], l_tigs[i])) {
                int str = card_strength(l_slots[i], l_tigs[i]);
                if (str < best_str) { best_str = str; best = l_acts[i]; }
            }
        }
        if (best >= 0) return best;
        /* No winner: play strongest */
        best = -1; int best_s = -1;
        for (int i = 0; i < nl; i++) {
            int str = card_strength(l_slots[i], l_tigs[i]);
            if (str > best_s) { best_s = str; best = l_acts[i]; }
        }
        return best;
    } else {
        /* Escape first (play slot 56-60) */
        for (int i = 0; i < nl; i++)
            if (slot_ctype(l_slots[i]) == CT_ESCAPE) return l_acts[i];
        /* Weakest non-winner */
        int best = -1, best_str = INT_MAX;
        for (int i = 0; i < nl; i++) {
            if (!would_beat(gs, l_slots[i], l_tigs[i])) {
                int str = card_strength(l_slots[i], l_tigs[i]);
                if (str < best_str) { best_str = str; best = l_acts[i]; }
            }
        }
        if (best >= 0) return best;
        /* All cards win: play weakest */
        best = -1; best_str = INT_MAX;
        for (int i = 0; i < nl; i++) {
            int str = card_strength(l_slots[i], l_tigs[i]);
            if (str < best_str) { best_str = str; best = l_acts[i]; }
        }
        return best;
    }
}

static int heuristic_action(const GameState *gs, int p) {
    (void)p;  /* gs_current_player is called internally */
    if (gs->phase == PHASE_BIDDING)
        return heuristic_bid(gs, gs_current_player(gs));
    return heuristic_play(gs, gs_current_player(gs));
}

/* ─── MLP forward pass ───────────────────────────────────────────────────── */

#if USE_AVX2
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(s);
    __m128 s2 = _mm_add_ps(s, sh);
    __m128 s3 = _mm_movehl_ps(sh, s2);
    return _mm_cvtss_f32(_mm_add_ss(s2, s3));
}

static void matvec_relu(const float *W, const float *b, const float *x,
                         float *out, int OUT, int IN)
{
    for (int i = 0; i < OUT; i++) {
        const float *row = W + (size_t)i * IN;
        __m256 acc = _mm256_setzero_ps();
        int j;
        for (j = 0; j + 8 <= IN; j += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(row+j), _mm256_loadu_ps(x+j), acc);
        float s = hsum256(acc) + b[i];
        for (; j < IN; j++) s += row[j] * x[j];
        out[i] = s > 0.0f ? s : 0.0f;
    }
}

static void matvec(const float *W, const float *b, const float *x,
                    float *out, int OUT, int IN)
{
    for (int i = 0; i < OUT; i++) {
        const float *row = W + (size_t)i * IN;
        __m256 acc = _mm256_setzero_ps();
        int j;
        for (j = 0; j + 8 <= IN; j += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(row+j), _mm256_loadu_ps(x+j), acc);
        float s = hsum256(acc) + b[i];
        for (; j < IN; j++) s += row[j] * x[j];
        out[i] = s;
    }
}
#else
static void matvec_relu(const float *W, const float *b, const float *x,
                         float *out, int OUT, int IN)
{
    for (int i = 0; i < OUT; i++) {
        const float *row = W + (size_t)i * IN;
        float s = b[i];
        for (int j = 0; j < IN; j++) s += row[j] * x[j];
        out[i] = s > 0.0f ? s : 0.0f;
    }
}
static void matvec(const float *W, const float *b, const float *x,
                    float *out, int OUT, int IN)
{
    for (int i = 0; i < OUT; i++) {
        const float *row = W + (size_t)i * IN;
        float s = b[i];
        for (int j = 0; j < IN; j++) s += row[j] * x[j];
        out[i] = s;
    }
}
#endif

/* obs must be MLP_IN floats (obs[244..247] = 0.0 padding) */
static void mlp_forward(const MLPWeights *w, const float obs[MLP_IN], float out[MLP_OUT]) {
    float h1[MLP_H1], h2[MLP_H2];
    matvec_relu((const float *)w->w1, w->b1, obs, h1, MLP_H1, MLP_IN);
    matvec_relu((const float *)w->w2, w->b2, h1,  h2, MLP_H2, MLP_H1);
    matvec     ((const float *)w->w3, w->b3, h2,  out, MLP_OUT, MLP_H2);
}

static void bid_mlp_forward(const float obs[MLP_IN], float out[BID_OUT]) {
    float h1[BID_H1], h2[BID_H2];
    matvec_relu((const float *)g_bid_adv.w1, g_bid_adv.b1, obs, h1, BID_H1, MLP_IN);
    matvec_relu((const float *)g_bid_adv.w2, g_bid_adv.b2, h1, h2, BID_H2, BID_H1);
    matvec     ((const float *)g_bid_adv.w3, g_bid_adv.b3, h2, out, BID_OUT, BID_H2);
}

static void play_mlp_forward(const float obs[MLP_IN], float out[PLAY_OUT]) {
    float h1[PLAY_H1], h2[PLAY_H2];
    matvec_relu((const float *)g_play_adv.w1, g_play_adv.b1, obs, h1, PLAY_H1, MLP_IN);
    matvec_relu((const float *)g_play_adv.w2, g_play_adv.b2, h1, h2, PLAY_H2, PLAY_H1);
    matvec     ((const float *)g_play_adv.w3, g_play_adv.b3, h2, out, PLAY_OUT, PLAY_H2);
}

/* Generic regret matching and sampling (variable action-space size N) */
static void regret_match_n(const float *adv, const uint8_t *mask, float *strat, int N) {
    float total = 0.0f;
    for (int a = 0; a < N; a++) {
        float v = (mask[a] && adv[a] > 0.0f) ? adv[a] : 0.0f;
        strat[a] = v;
        total += v;
    }
    if (total < 1e-12f) {
        int nl = 0;
        for (int a = 0; a < N; a++) nl += mask[a];
        float u = nl > 0 ? (1.0f / (float)nl) : 0.0f;
        for (int a = 0; a < N; a++) strat[a] = mask[a] ? u : 0.0f;
    } else {
        float inv = 1.0f / total;
        for (int a = 0; a < N; a++) strat[a] *= inv;
    }
}

static int sample_action_n(const float *strat, const uint8_t *mask, int N, uint64_t *rng) {
    float r = rng_float(rng);
    float cum = 0.0f;
    int last = -1;
    for (int a = 0; a < N; a++) {
        if (!mask[a]) continue;
        last = a;
        cum += strat[a];
        if (r < cum) return a;
    }
    return (last >= 0) ? last : 0;
}

/* ─── Regret matching ────────────────────────────────────────────────────── */

static void regret_match(const float adv[ACTION_SIZE], const uint8_t mask[ACTION_SIZE],
                          float strat[ACTION_SIZE])
{
    float total = 0.0f;
    for (int a = 0; a < ACTION_SIZE; a++) {
        float v = (mask[a] && adv[a] > 0.0f) ? adv[a] : 0.0f;
        strat[a] = v;
        total += v;
    }
    if (total < 1e-12f) {
        int nl = 0;
        for (int a = 0; a < ACTION_SIZE; a++) nl += mask[a];
        float u = nl > 0 ? (1.0f / (float)nl) : 0.0f;
        for (int a = 0; a < ACTION_SIZE; a++) strat[a] = mask[a] ? u : 0.0f;
    } else {
        float inv = 1.0f / total;
        for (int a = 0; a < ACTION_SIZE; a++) strat[a] *= inv;
    }
}

/* Sample action from strategy distribution */
static int sample_action(const float strat[ACTION_SIZE], const uint8_t mask[ACTION_SIZE],
                          uint64_t *rng)
{
    float r = rng_float(rng);
    float cum = 0.0f;
    int last = -1;
    for (int a = 0; a < ACTION_SIZE; a++) {
        if (!mask[a]) continue;
        last = a;
        cum += strat[a];
        if (r < cum) return a;
    }
    return (last >= 0) ? last : 0;
}

/* ─── Traversal buffers (static — safe with multiprocessing) ─────────────── */
#define MAX_DECISIONS  700   /* upper bound: 6 players × (10 bids + 10×10 plays) */
#define MAX_TRAV_DEC   200   /* traverser's decisions per game (1/n_players) */

static float   s_obs  [MAX_DECISIONS][MLP_IN];
static uint8_t s_masks[MAX_DECISIONS][ACTION_SIZE];
static float   s_strats[MAX_DECISIONS][ACTION_SIZE];
static int     s_n;

static int     t_indices[MAX_TRAV_DEC];
static int     t_actions[MAX_TRAV_DEC];
static float   t_adv_ests[MAX_TRAV_DEC][ACTION_SIZE];
static int     t_n;

/* ─── Split traversal buffers ────────────────────────────────────────────── */
static float   sp_bid_obs  [MAX_DECISIONS][MLP_IN];
static uint8_t sp_bid_masks[MAX_DECISIONS][BID_OUT];
static float   sp_bid_strats[MAX_DECISIONS][BID_OUT];
static int     sp_bid_n;

static float   sp_play_obs  [MAX_DECISIONS][MLP_IN];
static uint8_t sp_play_masks[MAX_DECISIONS][PLAY_OUT];
static float   sp_play_strats[MAX_DECISIONS][PLAY_OUT];
static int     sp_play_n;

static int   tp_bid_si  [MAX_TRAV_DEC];
static int   tp_bid_acts[MAX_TRAV_DEC];
static float tp_bid_adv [MAX_TRAV_DEC][BID_OUT];
static int   tp_bid_n;

static int   tp_play_si  [MAX_TRAV_DEC];
static int   tp_play_acts[MAX_TRAV_DEC];
static float tp_play_adv [MAX_TRAV_DEC][PLAY_OUT];
static int   tp_play_n;

static float sp_bid_adv_tgts [MAX_TRAV_DEC][BID_OUT];
static float sp_play_adv_tgts[MAX_TRAV_DEC][PLAY_OUT];

/* ─── Core traversal ─────────────────────────────────────────────────────── */

static int do_traverse(int traverser, uint64_t seed, int n_players,
                        float heuristic_frac,
                        /* outputs: */
                        float   **adv_obs_out,    /* [t_n][OBS_SIZE] */
                        uint8_t **adv_masks_out,  /* [t_n][ACTION_SIZE] */
                        float   **adv_tgt_out,    /* [t_n][ACTION_SIZE] */
                        int     **adv_acts_out,   /* [t_n] */
                        float   **strat_obs_out,
                        uint8_t **strat_masks_out,
                        float   **strat_strats_out,
                        int      *adv_n_out,
                        int      *strat_n_out)
{
    GameState gs;
    gs_init(&gs, n_players, seed);

    /* Second RNG stream for action-sampling (separate from game's deck RNG) */
    uint64_t rng = seed ^ 0xDEADBEEFCAFEBABEULL;

    s_n = 0; t_n = 0;

    while (gs.phase != PHASE_GAMEOVER) {
        int p = gs_current_player(&gs);

        /* Build observation + mask */
        if (s_n >= MAX_DECISIONS) break;  /* safety guard */
        gs_build_obs(&gs, p, s_obs[s_n]);
        gs_legal_mask(&gs, s_masks[s_n]);

        /* MLP forward → regret matching → strategy */
        float adv_est[ACTION_SIZE];
        mlp_forward(&g_adv, s_obs[s_n], adv_est);
        /* Zero illegal actions */
        for (int a = 0; a < ACTION_SIZE; a++)
            if (!s_masks[s_n][a]) adv_est[a] = 0.0f;
        regret_match(adv_est, s_masks[s_n], s_strats[s_n]);

        int action;
        if (p == traverser) {
            /* Record traverser decision */
            if (t_n < MAX_TRAV_DEC) {
                t_indices[t_n] = s_n;
                memcpy(t_adv_ests[t_n], adv_est, ACTION_SIZE * sizeof(float));
            }
            action = sample_action(s_strats[s_n], s_masks[s_n], &rng);
            if (t_n < MAX_TRAV_DEC) {
                t_actions[t_n] = action;
                t_n++;
            }
        } else {
            /* Opponent: mix heuristic */
            if (heuristic_frac > 0.0f && rng_float(&rng) < heuristic_frac)
                action = heuristic_action(&gs, p);
            else
                action = sample_action(s_strats[s_n], s_masks[s_n], &rng);
        }

        s_n++;
        gs_apply_action(&gs, action);
    }

    /* Compute utility */
    float utility = gs_utility(&gs, traverser);

    /* Compute advantage targets */
    static float adv_targets[MAX_TRAV_DEC][ACTION_SIZE];
    for (int k = 0; k < t_n; k++) {
        int idx = t_indices[k];
        memset(adv_targets[k], 0, ACTION_SIZE * sizeof(float));

        const float   *adv_est = t_adv_ests[k];
        const float   *strat   = s_strats[idx];
        const uint8_t *msk     = s_masks[idx];

        /* Baseline = expected advantage under current strategy */
        float baseline = 0.0f;
        for (int a = 0; a < ACTION_SIZE; a++)
            if (msk[a]) baseline += strat[a] * adv_est[a];

        /* Fill counterfactual estimates for ALL legal actions (Fix A) */
        for (int a = 0; a < ACTION_SIZE; a++) {
            if (msk[a]) adv_targets[k][a] = adv_est[a] - baseline;
        }
        /* Override taken action with IS-corrected utility signal (Fix B) */
        int a_taken    = t_actions[k];
        float prob     = strat[a_taken] > 0.05f ? strat[a_taken] : 0.05f;
        adv_targets[k][a_taken] = (utility - baseline) / prob;
    }

    /* Set output pointers */
    *adv_obs_out      = (t_n > 0) ? (float *)s_obs[t_indices[0]] : NULL;
    *adv_masks_out    = (t_n > 0) ? (uint8_t *)s_masks[t_indices[0]] : NULL;
    *adv_tgt_out      = (t_n > 0) ? (float *)adv_targets[0] : NULL;
    *adv_acts_out     = (t_n > 0) ? t_actions : NULL;
    *strat_obs_out    = (float *)s_obs;
    *strat_masks_out  = (uint8_t *)s_masks;
    *strat_strats_out = (float *)s_strats;
    *adv_n_out        = t_n;
    *strat_n_out      = s_n;
    return 0;
}

/* ─── Split traversal ────────────────────────────────────────────────────── */

static void do_traverse_split(int traverser, uint64_t seed, int n_players,
                               float heuristic_frac)
{
    GameState gs;
    gs_init(&gs, n_players, seed);
    uint64_t rng = seed ^ 0xDEADBEEFCAFEBABEULL;

    sp_bid_n = 0; sp_play_n = 0;
    tp_bid_n = 0; tp_play_n = 0;

    while (gs.phase != PHASE_GAMEOVER) {
        int p = gs_current_player(&gs);
        uint8_t full_mask[ACTION_SIZE];
        gs_legal_mask(&gs, full_mask);

        if (gs.phase == PHASE_BIDDING) {
            if (sp_bid_n >= MAX_DECISIONS) break;
            gs_build_obs(&gs, p, sp_bid_obs[sp_bid_n]);
            memcpy(sp_bid_masks[sp_bid_n], full_mask, BID_OUT);

            float adv_est[BID_OUT];
            bid_mlp_forward(sp_bid_obs[sp_bid_n], adv_est);
            for (int a = 0; a < BID_OUT; a++)
                if (!sp_bid_masks[sp_bid_n][a]) adv_est[a] = 0.0f;
            regret_match_n(adv_est, sp_bid_masks[sp_bid_n],
                           sp_bid_strats[sp_bid_n], BID_OUT);

            int abs_action;
            if (p == traverser && tp_bid_n < MAX_TRAV_DEC) {
                tp_bid_si[tp_bid_n] = sp_bid_n;
                memcpy(tp_bid_adv[tp_bid_n], adv_est, BID_OUT * sizeof(float));
                abs_action = sample_action_n(sp_bid_strats[sp_bid_n],
                                             sp_bid_masks[sp_bid_n], BID_OUT, &rng);
                tp_bid_acts[tp_bid_n] = abs_action; /* bid local == absolute */
                tp_bid_n++;
            } else if (p != traverser && heuristic_frac > 0.0f
                       && rng_float(&rng) < heuristic_frac) {
                abs_action = heuristic_action(&gs, p);
            } else {
                abs_action = sample_action_n(sp_bid_strats[sp_bid_n],
                                             sp_bid_masks[sp_bid_n], BID_OUT, &rng);
            }
            sp_bid_n++;
            gs_apply_action(&gs, abs_action);

        } else { /* PHASE_PLAYING */
            if (sp_play_n >= MAX_DECISIONS) break;
            gs_build_obs(&gs, p, sp_play_obs[sp_play_n]);

            /* Absolute → local play mask */
            int a;
            for (a = 0; a < 69; a++)
                sp_play_masks[sp_play_n][a] = full_mask[N_BID_ACTIONS + a];
            sp_play_masks[sp_play_n][69] = full_mask[ACT_TIG_ESCAPE];
            sp_play_masks[sp_play_n][70] = full_mask[ACT_TIG_PIRATE];

            float adv_est[PLAY_OUT];
            play_mlp_forward(sp_play_obs[sp_play_n], adv_est);
            for (a = 0; a < PLAY_OUT; a++)
                if (!sp_play_masks[sp_play_n][a]) adv_est[a] = 0.0f;
            regret_match_n(adv_est, sp_play_masks[sp_play_n],
                           sp_play_strats[sp_play_n], PLAY_OUT);

            int abs_action;
            if (p == traverser && tp_play_n < MAX_TRAV_DEC) {
                int local = sample_action_n(sp_play_strats[sp_play_n],
                                            sp_play_masks[sp_play_n], PLAY_OUT, &rng);
                tp_play_si  [tp_play_n] = sp_play_n;
                tp_play_acts[tp_play_n] = local;
                memcpy(tp_play_adv[tp_play_n], adv_est, PLAY_OUT * sizeof(float));
                tp_play_n++;
                abs_action = (local < 69) ? N_BID_ACTIONS + local
                           : (local == 69) ? ACT_TIG_ESCAPE : ACT_TIG_PIRATE;
            } else if (p != traverser && heuristic_frac > 0.0f
                       && rng_float(&rng) < heuristic_frac) {
                abs_action = heuristic_action(&gs, p);
            } else {
                int local = sample_action_n(sp_play_strats[sp_play_n],
                                            sp_play_masks[sp_play_n], PLAY_OUT, &rng);
                abs_action = (local < 69) ? N_BID_ACTIONS + local
                           : (local == 69) ? ACT_TIG_ESCAPE : ACT_TIG_PIRATE;
            }
            sp_play_n++;
            gs_apply_action(&gs, abs_action);
        }
    }

    float utility = gs_utility(&gs, traverser);

    /* Bid advantage targets */
    int k;
    for (k = 0; k < tp_bid_n; k++) {
        int idx = tp_bid_si[k];
        memset(sp_bid_adv_tgts[k], 0, BID_OUT * sizeof(float));
        const float   *ae  = tp_bid_adv[k];
        const float   *st  = sp_bid_strats[idx];
        const uint8_t *msk = sp_bid_masks[idx];
        float baseline = 0.0f;
        for (int a = 0; a < BID_OUT; a++)
            if (msk[a]) baseline += st[a] * ae[a];
        for (int a = 0; a < BID_OUT; a++)
            if (msk[a]) sp_bid_adv_tgts[k][a] = ae[a] - baseline;
        int at = tp_bid_acts[k];
        float prob = st[at] > 0.05f ? st[at] : 0.05f;
        sp_bid_adv_tgts[k][at] = (utility - baseline) / prob;
    }

    /* Play advantage targets */
    for (k = 0; k < tp_play_n; k++) {
        int idx = tp_play_si[k];
        memset(sp_play_adv_tgts[k], 0, PLAY_OUT * sizeof(float));
        const float   *ae  = tp_play_adv[k];
        const float   *st  = sp_play_strats[idx];
        const uint8_t *msk = sp_play_masks[idx];
        float baseline = 0.0f;
        for (int a = 0; a < PLAY_OUT; a++)
            if (msk[a]) baseline += st[a] * ae[a];
        for (int a = 0; a < PLAY_OUT; a++)
            if (msk[a]) sp_play_adv_tgts[k][a] = ae[a] - baseline;
        int at = tp_play_acts[k];
        float prob = st[at] > 0.05f ? st[at] : 0.05f;
        sp_play_adv_tgts[k][at] = (utility - baseline) / prob;
    }
}

/* ─── Python extension functions ─────────────────────────────────────────── */

/*
 * set_adv_weights(w1, b1, w2, b2, w3, b3)
 *
 * w1: float32 ndarray [512, 244]   (PyTorch Linear stores [out, in])
 * b1: float32 ndarray [512]
 * w2: float32 ndarray [512, 512]
 * b2: float32 ndarray [512]
 * w3: float32 ndarray [82, 512]
 * b3: float32 ndarray [82]
 *
 * w1 is padded to [512, 248] (MLP_IN=248) by zeroing columns 244-247.
 */
static PyObject *py_set_adv_weights(PyObject *self, PyObject *args) {
    PyObject *w1o, *b1o, *w2o, *b2o, *w3o, *b3o;
    if (!PyArg_ParseTuple(args, "OOOOOO", &w1o, &b1o, &w2o, &b2o, &w3o, &b3o))
        return NULL;

    Py_buffer w1b, b1b, w2b, b2b, w3b, b3b;
#define GET_BUF(obj, buf) \
    if (PyObject_GetBuffer(obj, &buf, PyBUF_SIMPLE|PyBUF_FORMAT) < 0) return NULL;

    GET_BUF(w1o, w1b); GET_BUF(b1o, b1b);
    GET_BUF(w2o, w2b); GET_BUF(b2o, b2b);
    GET_BUF(w3o, w3b); GET_BUF(b3o, b3b);
#undef GET_BUF

    /* Copy w1: [512][244] → [512][248] with zero padding */
    const float *src_w1 = (const float *)w1b.buf;
    for (int i = 0; i < MLP_H1; i++) {
        memcpy(g_adv.w1[i], src_w1 + (size_t)i * MLP_IN_RAW, MLP_IN_RAW * sizeof(float));
        memset(g_adv.w1[i] + MLP_IN_RAW, 0, (MLP_IN - MLP_IN_RAW) * sizeof(float));
    }
    memcpy(g_adv.b1, b1b.buf, MLP_H1 * sizeof(float));
    memcpy(g_adv.w2, w2b.buf, (size_t)MLP_H2 * MLP_H1 * sizeof(float));
    memcpy(g_adv.b2, b2b.buf, MLP_H2 * sizeof(float));
    memcpy(g_adv.w3, w3b.buf, (size_t)MLP_OUT * MLP_H2 * sizeof(float));
    memcpy(g_adv.b3, b3b.buf, MLP_OUT * sizeof(float));

    PyBuffer_Release(&w1b); PyBuffer_Release(&b1b);
    PyBuffer_Release(&w2b); PyBuffer_Release(&b2b);
    PyBuffer_Release(&w3b); PyBuffer_Release(&b3b);

    g_weights_loaded = 1;
    Py_RETURN_NONE;
}

/*
 * traverse(traverser, seed, n_players, heuristic_frac)
 *   → (adv_obs, adv_masks, adv_targets, adv_actions,
 *      strat_obs, strat_masks, strat_strategies)
 *
 * All returned arrays are newly allocated numpy arrays.
 * adv arrays have shape (n_adv, ...) where n_adv = traverser's decisions.
 * strat arrays have shape (n_strat, ...) where n_strat = all players' decisions.
 */
static PyObject *py_traverse(PyObject *self, PyObject *args) {
    int traverser, n_players;
    unsigned long long seed;
    float heuristic_frac = 0.4f;

    if (!PyArg_ParseTuple(args, "iKi|f", &traverser, &seed, &n_players, &heuristic_frac))
        return NULL;

    if (!g_weights_loaded) {
        PyErr_SetString(PyExc_RuntimeError,
            "skull_king_engine: adv weights not loaded (call set_adv_weights first)");
        return NULL;
    }

    float   *adv_obs, *adv_tgt, *strat_obs, *strat_strats;
    uint8_t *adv_masks, *strat_masks;
    int     *adv_acts;
    int adv_n, strat_n;

    do_traverse(traverser, (uint64_t)seed, n_players, heuristic_frac,
                &adv_obs, &adv_masks, &adv_tgt, &adv_acts,
                &strat_obs, &strat_masks, &strat_strats,
                &adv_n, &strat_n);

    npy_intp dim_obs[2]   = {adv_n,   OBS_SIZE};
    npy_intp dim_mask[2]  = {adv_n,   ACTION_SIZE};
    npy_intp dim_tgt[2]   = {adv_n,   ACTION_SIZE};
    npy_intp dim_acts[1]  = {adv_n};
    npy_intp dim_sobs[2]  = {strat_n, OBS_SIZE};
    npy_intp dim_smask[2] = {strat_n, ACTION_SIZE};
    npy_intp dim_sstrat[2]= {strat_n, ACTION_SIZE};

    /* Advantage arrays */
    PyObject *ao = PyArray_SimpleNew(2, dim_obs,  NPY_FLOAT);
    PyObject *am = PyArray_SimpleNew(2, dim_mask, NPY_BOOL);
    PyObject *at = PyArray_SimpleNew(2, dim_tgt,  NPY_FLOAT);
    PyObject *aa = PyArray_SimpleNew(1, dim_acts, NPY_INT64);
    /* Strategy arrays */
    PyObject *so = PyArray_SimpleNew(2, dim_sobs,  NPY_FLOAT);
    PyObject *sm = PyArray_SimpleNew(2, dim_smask, NPY_BOOL);
    PyObject *ss = PyArray_SimpleNew(2, dim_sstrat,NPY_FLOAT);

    if (!ao||!am||!at||!aa||!so||!sm||!ss) {
        Py_XDECREF(ao); Py_XDECREF(am); Py_XDECREF(at); Py_XDECREF(aa);
        Py_XDECREF(so); Py_XDECREF(sm); Py_XDECREF(ss);
        return NULL;
    }

    /* Copy adv data — obs only the OBS_SIZE columns (skip AVX padding) */
    if (adv_n > 0) {
        float   *ao_ptr = (float   *)PyArray_DATA((PyArrayObject *)ao);
        uint8_t *am_ptr = (uint8_t *)PyArray_DATA((PyArrayObject *)am);
        float   *at_ptr = (float   *)PyArray_DATA((PyArrayObject *)at);
        int64_t *aa_ptr = (int64_t *)PyArray_DATA((PyArrayObject *)aa);

        for (int k = 0; k < adv_n; k++) {
            int idx = t_indices[k];
            memcpy(ao_ptr + (size_t)k * OBS_SIZE, s_obs[idx],   OBS_SIZE * sizeof(float));
            memcpy(am_ptr + (size_t)k * ACTION_SIZE, s_masks[idx], ACTION_SIZE);
            memcpy(at_ptr + (size_t)k * ACTION_SIZE,
                   ((float *)adv_tgt) + (size_t)k * ACTION_SIZE,
                   ACTION_SIZE * sizeof(float));
            aa_ptr[k] = (int64_t)t_actions[k];
        }
    }

    /* Copy strategy data */
    if (strat_n > 0) {
        float   *so_ptr = (float   *)PyArray_DATA((PyArrayObject *)so);
        uint8_t *sm_ptr = (uint8_t *)PyArray_DATA((PyArrayObject *)sm);
        float   *ss_ptr = (float   *)PyArray_DATA((PyArrayObject *)ss);

        for (int k = 0; k < strat_n; k++) {
            memcpy(so_ptr + (size_t)k * OBS_SIZE, s_obs[k],    OBS_SIZE * sizeof(float));
            memcpy(sm_ptr + (size_t)k * ACTION_SIZE, s_masks[k],  ACTION_SIZE);
            memcpy(ss_ptr + (size_t)k * ACTION_SIZE, s_strats[k], ACTION_SIZE * sizeof(float));
        }
    }

    PyObject *result = PyTuple_Pack(7, ao, am, at, aa, so, sm, ss);
    Py_DECREF(ao); Py_DECREF(am); Py_DECREF(at); Py_DECREF(aa);
    Py_DECREF(so); Py_DECREF(sm); Py_DECREF(ss);
    return result;
}

static PyObject *py_set_bid_adv_weights(PyObject *self, PyObject *args) {
    PyObject *w1o, *b1o, *w2o, *b2o, *w3o, *b3o;
    if (!PyArg_ParseTuple(args, "OOOOOO", &w1o, &b1o, &w2o, &b2o, &w3o, &b3o))
        return NULL;
    Py_buffer w1b, b1b, w2b, b2b, w3b, b3b;
#define GET_BUF_BID(obj, buf) \
    if (PyObject_GetBuffer(obj, &buf, PyBUF_SIMPLE|PyBUF_FORMAT) < 0) return NULL;
    GET_BUF_BID(w1o,w1b); GET_BUF_BID(b1o,b1b);
    GET_BUF_BID(w2o,w2b); GET_BUF_BID(b2o,b2b);
    GET_BUF_BID(w3o,w3b); GET_BUF_BID(b3o,b3b);
#undef GET_BUF_BID
    const float *src_w1 = (const float *)w1b.buf;
    for (int i = 0; i < BID_H1; i++) {
        memcpy(g_bid_adv.w1[i], src_w1 + (size_t)i * MLP_IN_RAW, MLP_IN_RAW * sizeof(float));
        memset(g_bid_adv.w1[i] + MLP_IN_RAW, 0, (MLP_IN - MLP_IN_RAW) * sizeof(float));
    }
    memcpy(g_bid_adv.b1, b1b.buf, BID_H1 * sizeof(float));
    memcpy(g_bid_adv.w2, w2b.buf, (size_t)BID_H2 * BID_H1 * sizeof(float));
    memcpy(g_bid_adv.b2, b2b.buf, BID_H2 * sizeof(float));
    memcpy(g_bid_adv.w3, w3b.buf, (size_t)BID_OUT * BID_H2 * sizeof(float));
    memcpy(g_bid_adv.b3, b3b.buf, BID_OUT * sizeof(float));
    PyBuffer_Release(&w1b); PyBuffer_Release(&b1b);
    PyBuffer_Release(&w2b); PyBuffer_Release(&b2b);
    PyBuffer_Release(&w3b); PyBuffer_Release(&b3b);
    g_bid_loaded = 1;
    Py_RETURN_NONE;
}

static PyObject *py_set_play_adv_weights(PyObject *self, PyObject *args) {
    PyObject *w1o, *b1o, *w2o, *b2o, *w3o, *b3o;
    if (!PyArg_ParseTuple(args, "OOOOOO", &w1o, &b1o, &w2o, &b2o, &w3o, &b3o))
        return NULL;
    Py_buffer w1b, b1b, w2b, b2b, w3b, b3b;
#define GET_BUF_PLAY(obj, buf) \
    if (PyObject_GetBuffer(obj, &buf, PyBUF_SIMPLE|PyBUF_FORMAT) < 0) return NULL;
    GET_BUF_PLAY(w1o,w1b); GET_BUF_PLAY(b1o,b1b);
    GET_BUF_PLAY(w2o,w2b); GET_BUF_PLAY(b2o,b2b);
    GET_BUF_PLAY(w3o,w3b); GET_BUF_PLAY(b3o,b3b);
#undef GET_BUF_PLAY
    const float *src_w1 = (const float *)w1b.buf;
    for (int i = 0; i < PLAY_H1; i++) {
        memcpy(g_play_adv.w1[i], src_w1 + (size_t)i * MLP_IN_RAW, MLP_IN_RAW * sizeof(float));
        memset(g_play_adv.w1[i] + MLP_IN_RAW, 0, (MLP_IN - MLP_IN_RAW) * sizeof(float));
    }
    memcpy(g_play_adv.b1, b1b.buf, PLAY_H1 * sizeof(float));
    memcpy(g_play_adv.w2, w2b.buf, (size_t)PLAY_H2 * PLAY_H1 * sizeof(float));
    memcpy(g_play_adv.b2, b2b.buf, PLAY_H2 * sizeof(float));
    memcpy(g_play_adv.w3, w3b.buf, (size_t)PLAY_OUT * PLAY_H2 * sizeof(float));
    memcpy(g_play_adv.b3, b3b.buf, PLAY_OUT * sizeof(float));
    PyBuffer_Release(&w1b); PyBuffer_Release(&b1b);
    PyBuffer_Release(&w2b); PyBuffer_Release(&b2b);
    PyBuffer_Release(&w3b); PyBuffer_Release(&b3b);
    g_play_loaded = 1;
    Py_RETURN_NONE;
}

static PyObject *py_traverse_split(PyObject *self, PyObject *args) {
    int traverser, n_players;
    unsigned long long seed;
    float heuristic_frac = 0.4f;

    if (!PyArg_ParseTuple(args, "iKi|f", &traverser, &seed, &n_players, &heuristic_frac))
        return NULL;

    if (!g_bid_loaded || !g_play_loaded) {
        PyErr_SetString(PyExc_RuntimeError,
            "skull_king_engine: split weights not loaded "
            "(call set_bid_adv_weights and set_play_adv_weights first)");
        return NULL;
    }

    do_traverse_split(traverser, (uint64_t)seed, n_players, heuristic_frac);

    /* Allocate 14 output arrays */
    npy_intp d_ba_obs[2]  = {tp_bid_n,  OBS_SIZE};
    npy_intp d_ba_msk[2]  = {tp_bid_n,  BID_OUT};
    npy_intp d_ba_tgt[2]  = {tp_bid_n,  BID_OUT};
    npy_intp d_ba_act[1]  = {tp_bid_n};
    npy_intp d_bs_obs[2]  = {sp_bid_n,  OBS_SIZE};
    npy_intp d_bs_msk[2]  = {sp_bid_n,  BID_OUT};
    npy_intp d_bs_str[2]  = {sp_bid_n,  BID_OUT};
    npy_intp d_pa_obs[2]  = {tp_play_n, OBS_SIZE};
    npy_intp d_pa_msk[2]  = {tp_play_n, PLAY_OUT};
    npy_intp d_pa_tgt[2]  = {tp_play_n, PLAY_OUT};
    npy_intp d_pa_act[1]  = {tp_play_n};
    npy_intp d_ps_obs[2]  = {sp_play_n, OBS_SIZE};
    npy_intp d_ps_msk[2]  = {sp_play_n, PLAY_OUT};
    npy_intp d_ps_str[2]  = {sp_play_n, PLAY_OUT};

    PyObject *ba_obs = PyArray_SimpleNew(2, d_ba_obs, NPY_FLOAT);
    PyObject *ba_msk = PyArray_SimpleNew(2, d_ba_msk, NPY_BOOL);
    PyObject *ba_tgt = PyArray_SimpleNew(2, d_ba_tgt, NPY_FLOAT);
    PyObject *ba_act = PyArray_SimpleNew(1, d_ba_act, NPY_INT64);
    PyObject *bs_obs = PyArray_SimpleNew(2, d_bs_obs, NPY_FLOAT);
    PyObject *bs_msk = PyArray_SimpleNew(2, d_bs_msk, NPY_BOOL);
    PyObject *bs_str = PyArray_SimpleNew(2, d_bs_str, NPY_FLOAT);
    PyObject *pa_obs = PyArray_SimpleNew(2, d_pa_obs, NPY_FLOAT);
    PyObject *pa_msk = PyArray_SimpleNew(2, d_pa_msk, NPY_BOOL);
    PyObject *pa_tgt = PyArray_SimpleNew(2, d_pa_tgt, NPY_FLOAT);
    PyObject *pa_act = PyArray_SimpleNew(1, d_pa_act, NPY_INT64);
    PyObject *ps_obs = PyArray_SimpleNew(2, d_ps_obs, NPY_FLOAT);
    PyObject *ps_msk = PyArray_SimpleNew(2, d_ps_msk, NPY_BOOL);
    PyObject *ps_str = PyArray_SimpleNew(2, d_ps_str, NPY_FLOAT);

    if (!ba_obs||!ba_msk||!ba_tgt||!ba_act||!bs_obs||!bs_msk||!bs_str||
        !pa_obs||!pa_msk||!pa_tgt||!pa_act||!ps_obs||!ps_msk||!ps_str) {
        Py_XDECREF(ba_obs); Py_XDECREF(ba_msk); Py_XDECREF(ba_tgt); Py_XDECREF(ba_act);
        Py_XDECREF(bs_obs); Py_XDECREF(bs_msk); Py_XDECREF(bs_str);
        Py_XDECREF(pa_obs); Py_XDECREF(pa_msk); Py_XDECREF(pa_tgt); Py_XDECREF(pa_act);
        Py_XDECREF(ps_obs); Py_XDECREF(ps_msk); Py_XDECREF(ps_str);
        return NULL;
    }

    /* Copy bid advantage data */
    if (tp_bid_n > 0) {
        float   *ao = (float   *)PyArray_DATA((PyArrayObject *)ba_obs);
        uint8_t *am = (uint8_t *)PyArray_DATA((PyArrayObject *)ba_msk);
        float   *at = (float   *)PyArray_DATA((PyArrayObject *)ba_tgt);
        int64_t *aa = (int64_t *)PyArray_DATA((PyArrayObject *)ba_act);
        for (int k = 0; k < tp_bid_n; k++) {
            int idx = tp_bid_si[k];
            memcpy(ao + (size_t)k * OBS_SIZE, sp_bid_obs[idx],   OBS_SIZE * sizeof(float));
            memcpy(am + (size_t)k * BID_OUT,  sp_bid_masks[idx], BID_OUT);
            memcpy(at + (size_t)k * BID_OUT,  sp_bid_adv_tgts[k], BID_OUT * sizeof(float));
            aa[k] = (int64_t)tp_bid_acts[k];
        }
    }

    /* Copy bid strategy data */
    if (sp_bid_n > 0) {
        float   *so = (float   *)PyArray_DATA((PyArrayObject *)bs_obs);
        uint8_t *sm = (uint8_t *)PyArray_DATA((PyArrayObject *)bs_msk);
        float   *ss = (float   *)PyArray_DATA((PyArrayObject *)bs_str);
        for (int k = 0; k < sp_bid_n; k++) {
            memcpy(so + (size_t)k * OBS_SIZE, sp_bid_obs[k],    OBS_SIZE * sizeof(float));
            memcpy(sm + (size_t)k * BID_OUT,  sp_bid_masks[k],  BID_OUT);
            memcpy(ss + (size_t)k * BID_OUT,  sp_bid_strats[k], BID_OUT * sizeof(float));
        }
    }

    /* Copy play advantage data */
    if (tp_play_n > 0) {
        float   *ao = (float   *)PyArray_DATA((PyArrayObject *)pa_obs);
        uint8_t *am = (uint8_t *)PyArray_DATA((PyArrayObject *)pa_msk);
        float   *at = (float   *)PyArray_DATA((PyArrayObject *)pa_tgt);
        int64_t *aa = (int64_t *)PyArray_DATA((PyArrayObject *)pa_act);
        for (int k = 0; k < tp_play_n; k++) {
            int idx = tp_play_si[k];
            memcpy(ao + (size_t)k * OBS_SIZE,  sp_play_obs[idx],   OBS_SIZE * sizeof(float));
            memcpy(am + (size_t)k * PLAY_OUT,   sp_play_masks[idx], PLAY_OUT);
            memcpy(at + (size_t)k * PLAY_OUT,   sp_play_adv_tgts[k], PLAY_OUT * sizeof(float));
            aa[k] = (int64_t)tp_play_acts[k];
        }
    }

    /* Copy play strategy data */
    if (sp_play_n > 0) {
        float   *so = (float   *)PyArray_DATA((PyArrayObject *)ps_obs);
        uint8_t *sm = (uint8_t *)PyArray_DATA((PyArrayObject *)ps_msk);
        float   *ss = (float   *)PyArray_DATA((PyArrayObject *)ps_str);
        for (int k = 0; k < sp_play_n; k++) {
            memcpy(so + (size_t)k * OBS_SIZE,  sp_play_obs[k],    OBS_SIZE * sizeof(float));
            memcpy(sm + (size_t)k * PLAY_OUT,   sp_play_masks[k],  PLAY_OUT);
            memcpy(ss + (size_t)k * PLAY_OUT,   sp_play_strats[k], PLAY_OUT * sizeof(float));
        }
    }

    PyObject *result = PyTuple_Pack(14,
        ba_obs, ba_msk, ba_tgt, ba_act,
        bs_obs, bs_msk, bs_str,
        pa_obs, pa_msk, pa_tgt, pa_act,
        ps_obs, ps_msk, ps_str);
    Py_DECREF(ba_obs); Py_DECREF(ba_msk); Py_DECREF(ba_tgt); Py_DECREF(ba_act);
    Py_DECREF(bs_obs); Py_DECREF(bs_msk); Py_DECREF(bs_str);
    Py_DECREF(pa_obs); Py_DECREF(pa_msk); Py_DECREF(pa_tgt); Py_DECREF(pa_act);
    Py_DECREF(ps_obs); Py_DECREF(ps_msk); Py_DECREF(ps_str);
    return result;
}

/* ─── Module boilerplate ─────────────────────────────────────────────────── */

static PyMethodDef engine_methods[] = {
    {
        "set_adv_weights", py_set_adv_weights, METH_VARARGS,
        "set_adv_weights(w1,b1,w2,b2,w3,b3) — load float32 advantage-net weights"
    },
    {
        "traverse", py_traverse, METH_VARARGS,
        "traverse(traverser,seed,n_players,heuristic_frac=0.4) "
        "-> (adv_obs,adv_masks,adv_targets,adv_actions,strat_obs,strat_masks,strat_strategies)"
    },
    {
        "set_bid_adv_weights", py_set_bid_adv_weights, METH_VARARGS,
        "set_bid_adv_weights(w1,b1,w2,b2,w3,b3) — load float32 bid advantage-net weights"
    },
    {
        "set_play_adv_weights", py_set_play_adv_weights, METH_VARARGS,
        "set_play_adv_weights(w1,b1,w2,b2,w3,b3) — load float32 play advantage-net weights"
    },
    {
        "traverse_split", py_traverse_split, METH_VARARGS,
        "traverse_split(traverser,seed,n_players,heuristic_frac=0.4) "
        "-> 14-tuple of numpy arrays for split bid/play networks"
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef engine_module = {
    PyModuleDef_HEAD_INIT, "skull_king_engine", NULL, -1, engine_methods
};

PyMODINIT_FUNC PyInit_skull_king_engine(void) {
    import_array();
    return PyModule_Create(&engine_module);
}
