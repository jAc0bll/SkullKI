#pragma once
// C interface of the spot solver (see solver/src/spot_c_api.cpp).
#ifdef __cplusplus
extern "C" {
#endif
int sk_load(int round, const char* path);
const char* sk_spot(const char* text);
#ifdef __cplusplus
}
#endif
