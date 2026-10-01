#include "sk/solver/dcfr_update.hpp"

#include <cmath>

namespace sk::solver {

namespace {

void regretMatch(const std::vector<double>& regret, std::vector<double>& out) {
    double pos = 0.0;
    for (double r : regret) pos += (r > 0.0 ? r : 0.0);
    const std::size_t n = regret.size();
    for (std::size_t a = 0; a < n; ++a) {
        out[a] = (pos > 0.0) ? (regret[a] > 0.0 ? regret[a] / pos : 0.0) : 1.0 / n;
    }
}

} // namespace

void applyDCFRUpdate(InfosetTable& table, const std::vector<const DeltaMap*>& deltas,
                     int t_, const DCFRParams& params)
{
    const double t = static_cast<double>(t_);
    const double posDisc = std::pow(t, params.alpha) / (std::pow(t, params.alpha) + 1.0);
    const double negDisc = std::pow(t, params.beta)  / (std::pow(t, params.beta)  + 1.0);
    const double avgDisc = std::pow(t / (t + 1.0), params.gamma);

    table.forEach([&](const InfoKey&, InfoNode& n) {
        for (int a = 0; a < n.nA; ++a) n.stratSum[a] *= avgDisc;
    });
    for (const DeltaMap* dm : deltas) {
        for (const auto& [node, d] : *dm) {
            for (int a = 0; a < node->nA; ++a) {
                node->regret[a]   += d.dR[a];
                node->stratSum[a] += d.dS[a];
            }
        }
    }
    table.forEach([&](const InfoKey&, InfoNode& n) {
        for (int a = 0; a < n.nA; ++a) {
            double& r = n.regret[a];
            r *= (r > 0.0) ? posDisc : negDisc;
        }
        regretMatch(n.regret, n.current);
    });
}

} // namespace sk::solver
