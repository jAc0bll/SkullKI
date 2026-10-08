// Web: the browser renders the SVG itself (as an image), which handles every
// DiceBear style; react-native-svg's web renderer drops some masks.
import { memo } from 'react';
import { Image, View, type StyleProp, type ViewStyle } from 'react-native';
import { avatarSvg, type AvatarSpec, type Zoom } from './avatars';

export const Avatar = memo(function Avatar({
  avatar,
  size = 40,
  ring,
  style,
  zoom,
}: {
  avatar: AvatarSpec | null | undefined;
  zoom?: Zoom;
  size?: number;
  ring?: string;
  style?: StyleProp<ViewStyle>;
}) {
  const spec = avatar ?? { style: 'adventurer' as const, seed: '?' };
  const uri = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(avatarSvg(spec, zoom))}`;
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
      <Image source={{ uri }} style={{ width: '100%', height: '100%' }} />
    </View>
  );
});
