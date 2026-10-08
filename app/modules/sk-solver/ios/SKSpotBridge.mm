#import "SKSpotBridge.h"

#include "sk/solver/spot_c_api.h"

@implementation SKSpotBridge

+ (BOOL)loadRound:(int)round path:(NSString *)path {
  return sk_load(round, path.fileSystemRepresentation) != 0;
}

+ (NSString *)query:(NSString *)text {
  return [NSString stringWithUTF8String:sk_spot(text.UTF8String)];
}

@end
