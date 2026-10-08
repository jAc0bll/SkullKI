import { router } from 'expo-router';
import { useAccount } from '@/account/store';
import { GameTable } from '@/game/GameTable';
import { useLocalGame } from '@/game/useLocalGame';

// You against three bots, completely offline (the result goes to your
// profile when you are signed in and online).
export default function BotGame() {
  const { user, recordBotGame } = useAccount();
  const game = useLocalGame({ name: user?.name ?? 'Du', avatar: user?.avatar ?? null }, recordBotGame);
  return <GameTable game={game} title="Gegen die KI" onClose={() => router.back()} />;
}
