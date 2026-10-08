# The Skull King spot solver: the repo's C++ engine + solver, compiled into the
# app. scripts/sync-native.js copies the sources into ios/cpp (CocoaPods only
# takes files below the podspec); it runs on `npm install`.
Pod::Spec.new do |s|
  s.name           = 'SkSolver'
  s.version        = '1.0.0'
  s.summary        = 'Skull King GTO spot solver'
  s.description    = 'Skull King GTO spot solver (C++ engine and neural strategy)'
  s.license        = 'MIT'
  s.author         = 'SkullKI'
  s.homepage       = 'https://github.com/jAc0bll/SkullKI'
  s.platforms      = { :ios => '16.4' }
  s.source         = { git: 'https://github.com/jAc0bll/SkullKI.git' }
  s.static_framework = true

  s.dependency 'ExpoModulesCore'

  s.source_files = '*.{h,mm,swift}', 'cpp/**/*.{hpp,h,cpp}'
  s.public_header_files = 'SKSpotBridge.h'
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++20',
    'HEADER_SEARCH_PATHS' => '"$(PODS_TARGET_SRCROOT)/cpp/engine/include" "$(PODS_TARGET_SRCROOT)/cpp/solver/include"',
    'GCC_OPTIMIZATION_LEVEL' => '3',
  }
end
