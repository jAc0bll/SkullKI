/*
 * skull_king_core.c — C extension for hot-path Skull King game logic.
 *
 * Implements two functions called hundreds of thousands of times per CFR
 * iteration:
 *
 *   resolve_trick(cards_list)  →  (winner_player_index: int, bonus: int)
 *   legal_cards_mask(hand, played)  →  list[bool]
 *
 * Integer encodings (must match skull_king/trick.py):
 *
 *   Card type  : NUMBERED=0, ESCAPE=1, PIRATE=2, MERMAID=3, SKULL_KING=4, TIGRESS=5
 *   Suit       : BLACK=0 (trump), YELLOW=1, GREEN=2, PURPLE=3, NONE=-1
 *
 * Build (from skull_king/_core/):
 *   Windows MSVC : python setup.py build_ext --inplace
 *   Linux gcc    : python setup.py build_ext --inplace
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <limits.h>

/* ── Card type constants ─────────────────────────────────────────────── */
#define CT_NUMBERED    0
#define CT_ESCAPE      1
#define CT_PIRATE      2
#define CT_MERMAID     3
#define CT_SKULL_KING  4
#define CT_TIGRESS     5   /* should never appear as effective_type */

/* ── Suit constants ──────────────────────────────────────────────────── */
#define SUIT_BLACK    0    /* trump suit */
#define SUIT_NONE    -1

#define MAX_TRICK_CARDS  6
#define MAX_HAND_CARDS  14

typedef struct {
    int eff_type;       /* effective card type (Tigress resolved to PIRATE/ESCAPE) */
    int suit;           /* suit int, or SUIT_NONE */
    int value;          /* 1-14 for NUMBERED, 0 otherwise */
    int play_order;     /* 1-based; lower = played earlier */
    int player_index;
} CPlayedCard;

/* ── Internal helpers ────────────────────────────────────────────────── */

/* Return index (into cards[0..n]) of the earliest-played card with eff_type. */
static int first_of_type(const CPlayedCard *cards, int n, int eff_type) {
    int best_order = INT_MAX, best = -1;
    for (int i = 0; i < n; i++) {
        if (cards[i].eff_type == eff_type && cards[i].play_order < best_order) {
            best_order = cards[i].play_order;
            best = i;
        }
    }
    return best;
}

/* Resolve a trick that contains no SK / Pirate / Mermaid specials.
 * Returns index into cards[]. */
static int resolve_numbered(const CPlayedCard *cards, int n) {
    int i;

    /* All Escapes → trick leader wins (min play_order). */
    int all_escape = 1;
    for (i = 0; i < n; i++) {
        if (cards[i].eff_type != CT_ESCAPE) { all_escape = 0; break; }
    }
    if (all_escape) {
        int best = 0;
        for (i = 1; i < n; i++)
            if (cards[i].play_order < cards[best].play_order) best = i;
        return best;
    }

    /* Find led suit: suit of first NUMBERED card by play_order. */
    int led_suit = SUIT_NONE, first_order = INT_MAX;
    for (i = 0; i < n; i++) {
        if (cards[i].eff_type == CT_NUMBERED && cards[i].play_order < first_order) {
            first_order = cards[i].play_order;
            led_suit    = cards[i].suit;
        }
    }

    if (led_suit == SUIT_NONE) {
        /* Escape led the trick (spec §9.2).
         * Highest BLACK wins; otherwise first non-Escape colored card's suit
         * becomes an informal led suit. */
        int best_black = -1, best_black_val = -1;
        for (i = 0; i < n; i++) {
            if (cards[i].eff_type == CT_NUMBERED && cards[i].suit == SUIT_BLACK
                    && cards[i].value > best_black_val) {
                best_black_val = cards[i].value;
                best_black     = i;
            }
        }
        if (best_black >= 0) return best_black;

        /* Find first non-Escape colored card to establish informal led suit. */
        int first_colored = -1, fc_order = INT_MAX;
        for (i = 0; i < n; i++) {
            if (cards[i].eff_type != CT_ESCAPE && cards[i].play_order < fc_order) {
                fc_order     = cards[i].play_order;
                first_colored = i;
            }
        }
        led_suit = (first_colored >= 0) ? cards[first_colored].suit : SUIT_NONE;
    }

    /* BLACK (trump) beats all colored numbered cards (spec §4.3). */
    int best_black = -1, best_black_val = -1;
    for (i = 0; i < n; i++) {
        if (cards[i].eff_type == CT_NUMBERED && cards[i].suit == SUIT_BLACK
                && cards[i].value > best_black_val) {
            best_black_val = cards[i].value;
            best_black     = i;
        }
    }
    if (best_black >= 0) return best_black;

    /* Highest card of led suit wins (spec §5 step 5b). */
    int best_led = -1, best_led_val = -1;
    for (i = 0; i < n; i++) {
        if (cards[i].eff_type == CT_NUMBERED && cards[i].suit == led_suit
                && cards[i].value > best_led_val) {
            best_led_val = cards[i].value;
            best_led     = i;
        }
    }
    return best_led;   /* guaranteed non-negative: at least one non-Escape card */
}

/* Compute bonus points for the trick winner (spec §6.2–6.3). */
static int compute_bonus(const CPlayedCard *cards, int n, int winner_idx) {
    int bonus = 0, i;
    int wtype = cards[winner_idx].eff_type;

    /* Black-14 bonus (+20): awarded when a special card (SK/Mermaid/Pirate) wins
     * and a Black-14 numbered card is in the trick. */
    if (wtype == CT_SKULL_KING || wtype == CT_MERMAID || wtype == CT_PIRATE) {
        for (i = 0; i < n; i++) {
            if (cards[i].eff_type == CT_NUMBERED
                    && cards[i].suit  == SUIT_BLACK
                    && cards[i].value == 14) {
                bonus += 20;
                break;
            }
        }
    }

    /* SK captures Pirates: +30 each (spec §6.2). */
    if (wtype == CT_SKULL_KING) {
        for (i = 0; i < n; i++)
            if (cards[i].eff_type == CT_PIRATE) bonus += 30;
    }

    /* Mermaid captures SK: +40 (spec §6.2). */
    if (wtype == CT_MERMAID) {
        for (i = 0; i < n; i++) {
            if (cards[i].eff_type == CT_SKULL_KING) { bonus += 40; break; }
        }
    }

    return bonus;
}

/* ── Python-facing functions ─────────────────────────────────────────── */

/*
 * resolve_trick(cards_list) -> (winner_player_index: int, bonus: int)
 *
 * cards_list: list of 5-tuples
 *   (eff_type_int, suit_int, value_int, play_order_int, player_index_int)
 * eff_type uses CT_* constants; Tigress must already be resolved.
 */
static PyObject *py_resolve_trick(PyObject *self, PyObject *args) {
    PyObject *cards_list;
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &cards_list))
        return NULL;

    Py_ssize_t n = PyList_GET_SIZE(cards_list);
    if (n == 0) {
        PyErr_SetString(PyExc_ValueError, "Cannot resolve empty trick");
        return NULL;
    }
    if (n > MAX_TRICK_CARDS) {
        PyErr_SetString(PyExc_ValueError, "Too many cards in trick (max 6)");
        return NULL;
    }

    CPlayedCard cards[MAX_TRICK_CARDS];
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PyList_GET_ITEM(cards_list, i);
        if (!PyTuple_Check(item) || PyTuple_GET_SIZE(item) != 5) {
            PyErr_SetString(PyExc_TypeError,
                            "Each card must be a 5-tuple "
                            "(eff_type, suit, value, play_order, player_index)");
            return NULL;
        }
        cards[i].eff_type     = (int)PyLong_AsLong(PyTuple_GET_ITEM(item, 0));
        cards[i].suit         = (int)PyLong_AsLong(PyTuple_GET_ITEM(item, 1));
        cards[i].value        = (int)PyLong_AsLong(PyTuple_GET_ITEM(item, 2));
        cards[i].play_order   = (int)PyLong_AsLong(PyTuple_GET_ITEM(item, 3));
        cards[i].player_index = (int)PyLong_AsLong(PyTuple_GET_ITEM(item, 4));
        if (PyErr_Occurred()) return NULL;
    }

    /* Determine winner following spec §5 priority order. */
    int has_sk = 0, has_mermaid = 0, has_pirate = 0;
    for (int i = 0; i < (int)n; i++) {
        if (cards[i].eff_type == CT_SKULL_KING) has_sk      = 1;
        if (cards[i].eff_type == CT_MERMAID)    has_mermaid = 1;
        if (cards[i].eff_type == CT_PIRATE)     has_pirate  = 1;
    }

    int winner_idx;
    if (has_sk && has_mermaid && !has_pirate)
        winner_idx = first_of_type(cards, (int)n, CT_MERMAID);
    else if (has_sk)
        winner_idx = first_of_type(cards, (int)n, CT_SKULL_KING);
    else if (has_pirate)
        winner_idx = first_of_type(cards, (int)n, CT_PIRATE);
    else if (has_mermaid)
        winner_idx = first_of_type(cards, (int)n, CT_MERMAID);
    else
        winner_idx = resolve_numbered(cards, (int)n);

    if (winner_idx < 0) {
        PyErr_SetString(PyExc_RuntimeError,
                        "resolve_trick: no winner found (malformed input?)");
        return NULL;
    }

    int bonus = compute_bonus(cards, (int)n, winner_idx);
    return Py_BuildValue("(ii)", cards[winner_idx].player_index, bonus);
}

/*
 * legal_cards_mask(hand_list, played_list) -> list[bool]
 *
 * hand_list  : list of 2-tuples (card_type_int, suit_int)
 *   card_type_int = 0 (CT_NUMBERED) or anything else (special — always legal).
 *
 * played_list: list of 3-tuples (card_type_int, suit_int, play_order_int)
 *   Used only to determine the led suit (first NUMBERED card by play_order).
 *
 * Returns a Python list of bool, one per hand card.
 */
static PyObject *py_legal_cards_mask(PyObject *self, PyObject *args) {
    PyObject *hand_list, *played_list;
    if (!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &hand_list,
                                        &PyList_Type, &played_list))
        return NULL;

    Py_ssize_t hand_n   = PyList_GET_SIZE(hand_list);
    Py_ssize_t played_n = PyList_GET_SIZE(played_list);

    if (hand_n > MAX_HAND_CARDS) {
        PyErr_SetString(PyExc_ValueError, "Hand too large (max 14 cards)");
        return NULL;
    }

    /* Parse hand: (card_type_int, suit_int). */
    int hand_types[MAX_HAND_CARDS], hand_suits[MAX_HAND_CARDS];
    for (Py_ssize_t i = 0; i < hand_n; i++) {
        PyObject *item = PyList_GET_ITEM(hand_list, i);
        hand_types[i] = (int)PyLong_AsLong(PyTuple_GET_ITEM(item, 0));
        hand_suits[i] = (int)PyLong_AsLong(PyTuple_GET_ITEM(item, 1));
        if (PyErr_Occurred()) return NULL;
    }

    /* Find led suit: suit of the first NUMBERED card by play_order. */
    int led_suit = SUIT_NONE, first_order = INT_MAX;
    for (Py_ssize_t i = 0; i < played_n; i++) {
        PyObject *item  = PyList_GET_ITEM(played_list, i);
        int ptype  = (int)PyLong_AsLong(PyTuple_GET_ITEM(item, 0));
        int psuit  = (int)PyLong_AsLong(PyTuple_GET_ITEM(item, 1));
        int porder = (int)PyLong_AsLong(PyTuple_GET_ITEM(item, 2));
        if (PyErr_Occurred()) return NULL;
        if (ptype == CT_NUMBERED && porder < first_order) {
            first_order = porder;
            led_suit    = psuit;
        }
    }

    PyObject *mask = PyList_New(hand_n);
    if (!mask) return NULL;

    if (led_suit == SUIT_NONE) {
        /* No led suit yet: every card is legal. */
        for (Py_ssize_t i = 0; i < hand_n; i++) {
            Py_INCREF(Py_True);
            PyList_SET_ITEM(mask, i, Py_True);
        }
        return mask;
    }

    /* Check whether the player holds any card of led suit. */
    int has_led = 0;
    for (Py_ssize_t i = 0; i < hand_n; i++) {
        if (hand_types[i] == CT_NUMBERED && hand_suits[i] == led_suit) {
            has_led = 1; break;
        }
    }

    if (!has_led) {
        /* Void in led suit: every card is legal (suit-following waived). */
        for (Py_ssize_t i = 0; i < hand_n; i++) {
            Py_INCREF(Py_True);
            PyList_SET_ITEM(mask, i, Py_True);
        }
    } else {
        /* Must follow: specials (non-NUMBERED) and led-suit cards are legal. */
        for (Py_ssize_t i = 0; i < hand_n; i++) {
            int legal = (hand_types[i] != CT_NUMBERED)
                     || (hand_suits[i] == led_suit);
            PyObject *b = legal ? Py_True : Py_False;
            Py_INCREF(b);
            PyList_SET_ITEM(mask, i, b);
        }
    }
    return mask;
}

/* ── Module boilerplate ──────────────────────────────────────────────── */

static PyMethodDef skull_king_methods[] = {
    {
        "resolve_trick",
        py_resolve_trick,
        METH_VARARGS,
        "resolve_trick(cards) -> (winner_player_index, bonus_points)\n"
        "cards: list of 5-tuples (eff_type, suit, value, play_order, player_index)"
    },
    {
        "legal_cards_mask",
        py_legal_cards_mask,
        METH_VARARGS,
        "legal_cards_mask(hand, played) -> list[bool]\n"
        "hand: list of (card_type_int, suit_int) tuples\n"
        "played: list of (card_type_int, suit_int, play_order) tuples"
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef skull_king_module = {
    PyModuleDef_HEAD_INIT,
    "skull_king_core",      /* module name */
    NULL,                   /* docstring */
    -1,                     /* per-interpreter state size */
    skull_king_methods,
};

PyMODINIT_FUNC PyInit_skull_king_core(void) {
    return PyModule_Create(&skull_king_module);
}
