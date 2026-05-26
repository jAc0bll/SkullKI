// Python bindings for the Skull King engine.
//
// Module name: `skullking` (built target: skullking.<so/pyd>).
// Build via the top-level CMake with -DSK_BUILD_PYTHON=ON.
//
// Exposes: enums (Suit, Phase, ActionType), Action (factory + getters),
// GameState (mostly read-only views), Observation, Rng (wrapping mt19937_64),
// the engine entry points (initial_state, deal_round, legal_actions,
// apply_action, step, observe), encode→numpy, and the three concrete agents.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "sk/cards.hpp"
#include "sk/card_set.hpp"
#include "sk/state.hpp"
#include "sk/action.hpp"
#include "sk/rules.hpp"
#include "sk/scoring.hpp"
#include "sk/game.hpp"
#include "sk/observation.hpp"
#include "sk/encoder.hpp"
#include "sk/action_index.hpp"
#include "sk/agent.hpp"
#include "sk/random_agent.hpp"
#include "sk/heuristic_agent.hpp"
#include "sk/ismcts.hpp"

#ifdef SK_HAS_TORCH
#include "sk/torch_model.hpp"
#include "sk/torch_nn_agent.hpp"
#include "sk/belief_model.hpp"
#include "sk/neural_mcts.hpp"
#endif

#include <random>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

template <std::size_t N>
std::vector<int> intArray(const std::int8_t (&arr)[N]) {
    std::vector<int> v;
    v.reserve(N);
    for (std::size_t i = 0; i < N; ++i) v.push_back(arr[i]);
    return v;
}

std::vector<int> cardsOf(const sk::CardSet& cs) {
    std::vector<int> v;
    cs.forEach([&](sk::Card c) { v.push_back(c); });
    return v;
}

py::array_t<float> encodeToNumpy(const sk::Observation& obs) {
    std::vector<float> v = sk::encode(obs);
    py::array_t<float> arr(static_cast<py::ssize_t>(v.size()));
    std::memcpy(arr.mutable_data(), v.data(), v.size() * sizeof(float));
    return arr;
}

} // namespace

PYBIND11_MODULE(skullking, m) {
    m.doc() = "Skull King game engine + agents (C++ core, Python bindings).";

    // ---- Module-level constants ----
    m.attr("N_CARDS")     = sk::N_CARDS;
    m.attr("N_PLAYERS")   = sk::N_PLAYERS;
    m.attr("MAX_ROUND")   = sk::MAX_ROUND;
    m.attr("ENC_DIM")     = sk::ENC_DIM;
    m.attr("ACTION_DIM")  = sk::ACTION_DIM;
    m.attr("ESCAPE_BASE") = static_cast<int>(sk::ESCAPE_OFFSET);
    m.attr("MERMAID_BASE") = static_cast<int>(sk::MERMAID_OFFSET);
    m.attr("PIRATE_BASE")  = static_cast<int>(sk::PIRATE_OFFSET);
    m.attr("TIGRESS")     = static_cast<int>(sk::TIGRESS);
    m.attr("SKULL_KING")  = static_cast<int>(sk::SKULL_KING);

    // ---- Enums ----
    py::enum_<sk::Suit>(m, "Suit")
        .value("Yellow", sk::Suit::Yellow)
        .value("Green",  sk::Suit::Green)
        .value("Purple", sk::Suit::Purple)
        .value("Black",  sk::Suit::Black)
        .value("NONE",   sk::Suit::None);

    py::enum_<sk::Phase>(m, "Phase")
        .value("Bidding", sk::Phase::Bidding)
        .value("Playing", sk::Phase::Playing)
        .value("GameEnd", sk::Phase::GameEnd);

    py::enum_<sk::ActionType>(m, "ActionType")
        .value("Bid",         sk::ActionType::Bid)
        .value("Play",        sk::ActionType::Play)
        .value("TigressMode", sk::ActionType::TigressMode);

    // ---- Card helpers ----
    m.def("card_name", &sk::cardName);
    m.def("suit_of",   &sk::suitOf);
    m.def("value_of",  &sk::valueOf);
    m.def("is_colored", &sk::isColored);
    m.def("is_special", &sk::isSpecial);
    m.def("is_trump",   &sk::isTrump);
    m.def("is_escape",  &sk::isEscape);
    m.def("is_pirate",  &sk::isPirate);
    m.def("is_mermaid", &sk::isMermaid);
    m.def("is_tigress", &sk::isTigress);
    m.def("is_skull_king", &sk::isSkullKing);

    // ---- Action ----
    py::class_<sk::Action>(m, "Action")
        .def_static("bid",          &sk::Action::makeBid,         py::arg("value"))
        .def_static("play",         &sk::Action::makePlay,        py::arg("card"))
        .def_static("tigress_mode", &sk::Action::makeTigressMode, py::arg("as_pirate"))
        .def_readonly("type", &sk::Action::type)
        .def_property_readonly("bid_value", [](const sk::Action& a) {
            if (a.type != sk::ActionType::Bid) throw py::value_error("not a Bid action");
            return static_cast<int>(a.bid);
        })
        .def_property_readonly("card", [](const sk::Action& a) {
            if (a.type != sk::ActionType::Play) throw py::value_error("not a Play action");
            return static_cast<int>(a.card);
        })
        .def_property_readonly("as_pirate", [](const sk::Action& a) {
            if (a.type != sk::ActionType::TigressMode)
                throw py::value_error("not a TigressMode action");
            return a.asPirate;
        })
        .def("__eq__", [](const sk::Action& a, const sk::Action& b) { return a == b; })
        .def("__repr__", [](const sk::Action& a) -> std::string {
            switch (a.type) {
                case sk::ActionType::Bid:
                    return "Action.bid(" + std::to_string(a.bid) + ")";
                case sk::ActionType::Play:
                    return "Action.play(" + std::to_string(a.card) +
                           " [" + sk::cardName(a.card) + "])";
                case sk::ActionType::TigressMode:
                    return std::string("Action.tigress_mode(") +
                           (a.asPirate ? "pirate" : "escape") + ")";
            }
            return "Action(?)";
        });

    // ---- GameState (mostly read-only views) ----
    py::class_<sk::GameState>(m, "GameState")
        .def(py::init<>())
        .def_readwrite("round_number",     &sk::GameState::roundNumber)
        .def_readwrite("phase",            &sk::GameState::phase)
        .def_readwrite("current_player",   &sk::GameState::currentPlayer)
        .def_readwrite("start_player",     &sk::GameState::startPlayer)
        .def_readwrite("trick_leader",     &sk::GameState::trickLeader)
        .def_readwrite("trick_size",       &sk::GameState::trickSize)
        .def_readwrite("tricks_played",    &sk::GameState::tricksPlayed)
        .def_readwrite("lead_suit",        &sk::GameState::leadSuit)
        .def_readwrite("free_trick",       &sk::GameState::freeTrick)
        .def_readwrite("pending_tigress",  &sk::GameState::pendingTigress)
        .def_readwrite("tigress_as_pirate",&sk::GameState::tigressAsPirate)
        .def_readwrite("bids_submitted",   &sk::GameState::bidsSubmitted)
        .def_property_readonly("bids",       [](const sk::GameState& s) { return intArray(s.bids); })
        .def_property_readonly("tricks_won", [](const sk::GameState& s) { return intArray(s.tricksWon); })
        .def_property_readonly("scores", [](const sk::GameState& s) {
            std::vector<int> v(s.scores, s.scores + sk::N_PLAYERS); return v;
        })
        .def_property_readonly("hands", [](const sk::GameState& s) {
            std::vector<std::vector<int>> out(sk::N_PLAYERS);
            for (int p = 0; p < sk::N_PLAYERS; ++p) out[p] = cardsOf(s.hands[p]);
            return out;
        })
        .def_property_readonly("trick_cards", [](const sk::GameState& s) {
            std::vector<int> v;
            for (int i = 0; i < s.trickSize; ++i) v.push_back(s.trickCards[i]);
            return v;
        })
        .def_property_readonly("trick_players", [](const sk::GameState& s) {
            std::vector<int> v;
            for (int i = 0; i < s.trickSize; ++i) v.push_back(s.trickPlayers[i]);
            return v;
        })
        .def_property_readonly("captured", [](const sk::GameState& s) {
            std::vector<std::vector<int>> out(sk::N_PLAYERS);
            for (int p = 0; p < sk::N_PLAYERS; ++p) out[p] = cardsOf(s.captured[p]);
            return out;
        });

    // ---- Observation ----
    py::class_<sk::Observation>(m, "Observation")
        .def_readwrite("perspective", &sk::Observation::perspective)
        .def_property_readonly("state",
            [](sk::Observation& o) -> sk::GameState& { return o.s; },
            py::return_value_policy::reference_internal)
        .def_property_readonly("hand_sizes", [](const sk::Observation& o) {
            std::vector<int> v(o.handSizes.begin(), o.handSizes.end()); return v;
        })
        .def_property_readonly("own_hand", [](const sk::Observation& o) {
            return cardsOf(o.s.hands[o.perspective]);
        });

    // ---- Rng wrapper ----
    py::class_<std::mt19937_64>(m, "Rng")
        .def(py::init<std::uint64_t>(), py::arg("seed") = 0ULL)
        .def("seed", [](std::mt19937_64& r, std::uint64_t s) { r.seed(s); }, py::arg("seed"));

    // ---- Engine entry points ----
    m.def("initial_state", &sk::initialState, py::arg("start_player") = 0);
    m.def("legal_actions", &sk::legalActions);
    m.def("apply_action", [](sk::GameState& s, const sk::Action& a) { sk::applyAction(s, a); });
    m.def("deal_round",   [](sk::GameState& s, std::mt19937_64& rng) { sk::dealRound(s, rng); });
    m.def("step",         [](sk::GameState& s, const sk::Action& a, std::mt19937_64& rng) {
        sk::step(s, a, rng);
    });
    m.def("is_terminal",  &sk::isTerminal);
    m.def("observe",      &sk::observe);
    m.def("encode",       &encodeToNumpy);
    m.def("action_to_index", &sk::actionToIndex);
    m.def("index_to_action", &sk::indexToAction);

    // ---- Agents ----
    py::class_<sk::Agent>(m, "Agent")
        .def("select_action",
             [](sk::Agent& agent, const sk::GameState& s, std::mt19937_64& rng) {
                 // selectAction takes a non-const ref; make a copy so Python isn't surprised.
                 sk::GameState copy = s;
                 return agent.selectAction(copy, rng);
             },
             py::arg("state"), py::arg("rng"))
        .def_property_readonly("name", [](const sk::Agent& a) { return std::string(a.name()); });

    py::class_<sk::RandomAgent,    sk::Agent>(m, "RandomAgent").def(py::init<>());
    py::class_<sk::HeuristicAgent, sk::Agent>(m, "HeuristicAgent").def(py::init<>());

    py::class_<sk::ISMCTSConfig>(m, "ISMCTSConfig")
        .def(py::init<>())
        .def_readwrite("num_simulations", &sk::ISMCTSConfig::numSimulations)
        .def_readwrite("exploration_c",   &sk::ISMCTSConfig::explorationC)
        .def_readwrite("reward_scale",    &sk::ISMCTSConfig::rewardScale);

    py::class_<sk::ISMCTSResult>(m, "ISMCTSResult")
        .def_readonly("action", &sk::ISMCTSResult::action)
        .def_property_readonly("values", [](const sk::ISMCTSResult& r) {
            std::vector<double> v(r.values.begin(), r.values.end());
            return v;
        })
        .def_property_readonly("root_actions", [](const sk::ISMCTSResult& r) {
            return r.rootActions;
        })
        .def_property_readonly("visits", [](const sk::ISMCTSResult& r) {
            return r.visits;
        });

    py::class_<sk::ISMCTSAgent, sk::Agent>(m, "ISMCTSAgent")
        .def(py::init<sk::ISMCTSConfig>(), py::arg("config") = sk::ISMCTSConfig{})
        .def("select_action_with_targets",
             [](sk::ISMCTSAgent& a, const sk::GameState& s, std::mt19937_64& rng) {
                 sk::GameState copy = s;
                 return a.selectActionWithTargets(copy, rng);
             },
             py::arg("state"), py::arg("rng"));

#ifdef SK_HAS_TORCH
    // ---- Torch model evaluators ----
    py::class_<sk::TorchModelEvaluator>(m, "TorchModelEvaluator")
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("path"), py::arg("device") = "cpu");

    py::class_<sk::BeliefEvaluator>(m, "BeliefEvaluator")
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("path"), py::arg("device") = "cpu");

    py::class_<sk::TorchNNAgent, sk::Agent>(m, "TorchNNAgent")
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("path"), py::arg("device") = "cpu");

    py::class_<sk::NeuralMCTSConfig>(m, "NeuralMCTSConfig")
        .def(py::init<>())
        .def_readwrite("mcts",         &sk::NeuralMCTSConfig::mcts)
        .def_readwrite("use_nn_value", &sk::NeuralMCTSConfig::useNNValue)
        .def("set_belief",
             [](sk::NeuralMCTSConfig& cfg, sk::BeliefEvaluator* b) { cfg.belief = b; },
             py::arg("belief").none(true));

    py::class_<sk::NeuralMCTSAgent, sk::Agent>(m, "NeuralMCTSAgent")
        .def(py::init<sk::TorchModelEvaluator*, sk::NeuralMCTSConfig>(),
             py::arg("evaluator"), py::arg("config") = sk::NeuralMCTSConfig{},
             py::keep_alive<1, 2>())   // keep evaluator alive
        .def("select_action_with_targets",
             [](sk::NeuralMCTSAgent& a, const sk::GameState& s, std::mt19937_64& rng) {
                 sk::GameState copy = s;
                 return a.selectActionWithTargets(copy, rng);
             },
             py::arg("state"), py::arg("rng"));

    m.attr("HAS_TORCH") = true;
#else
    m.attr("HAS_TORCH") = false;
#endif
}
