import { router } from 'expo-router';
import { GameTable } from '@/game/GameTable';
import { useLocalGame } from '@/game/useLocalGame';

// You against three bots, completely offline.
export default function BotGame() {
  const game = useLocalGame();
  return <GameTable game={game} title="Gegen die KI" onClose={() => router.back()} />;
}
