import { memo } from 'react';
import { View, type StyleProp, type ViewStyle } from 'react-native';
import { SvgXml } from 'react-native-svg';
import { avatarSvg, type AvatarSpec } from './avatars';

export const Avatar = memo(function Avatar({
  avatar,
  size = 40,
  ring,
  style,
}: {
  avatar: AvatarSpec | null | undefined;
  size?: number;
  ring?: string;
  style?: StyleProp<ViewStyle>;
}) {
  const spec = avatar ?? { style: 'adventurer' as const, seed: '?' };
  return (
    <View
      style={[
        {
          width: size,
          height: size,
          borderRadius: size / 2,
          overflow: 'hidden',
          backgroundColor: 'rgba(255,255,255,0.08)',
          borderWidth: ring ? 2 : 0,
          borderColor: ring,
        },
        style,
      ]}>
      <SvgXml xml={avatarSvg(spec)} width="100%" height="100%" />
    </View>
  );
});
