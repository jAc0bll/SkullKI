#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Objective-C face of the C++ spot solver (solver/src/spot_c_api.cpp).
@interface SKSpotBridge : NSObject
+ (BOOL)loadRound:(int)round path:(NSString *)path;
+ (NSString *)query:(NSString *)text;
@end

NS_ASSUME_NONNULL_END
