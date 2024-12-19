import core.Constants;
import core.Types;
import core.game.Game;
import gui.GUI;
import gui.WindowInput;
import org.json.JSONArray;
import org.json.JSONObject;
import players.*;
import players.emcts.EMCTSAgent;
import players.emcts.EMCTSParams;
import players.mc.MCParams;
import players.mc.MonteCarloAgent;
import players.mcts.MCTSParams;
import players.mcts.MCTSPlayer;
import players.oep.OEPAgent;
import players.oep.OEPParams;
import players.osla.OSLAParams;
import players.osla.OneStepLookAheadAgent;
import players.portfolio.SimplePortfolio;
import players.portfolioMCTS.PortfolioMCTSParams;
import players.portfolioMCTS.PortfolioMCTSPlayer;
import players.rhea.RHEAAgent;
import players.rhea.RHEAParams;
import players.portfolio.RandomPortfolio;
import utils.file.IO;
import me.tongfei.progressbar.*;

import java.io.File;
import java.util.*;

import static core.Types.GAME_MODE.*;

/**
 * Entry point of the framework.
 */
public class Play {

    private static boolean RUN_VERBOSE = true;
    private static long AGENT_SEED = -1;
    private static long GAME_SEED = -1;

    public static void main(String[] args) {
        try {
            JSONObject config = new IO().readJSON("play.json");

            if (config != null && !config.isEmpty()) {
                String runMode = config.getString("Run Mode");
                Constants.VERBOSE = config.getBoolean("Verbose");
                RUN_VERBOSE = config.getBoolean("Verbose");

                int numGames = config.getInt("Number of Games");
                if (runMode.equalsIgnoreCase("Replay")) {
                    numGames = 1;
                }
                System.out.println("Starting " + numGames + " games with run mode: " + runMode);

                try (ProgressBar pb = new ProgressBar("Games", numGames)) {
                    for (int gameIndex = 0; gameIndex < numGames; gameIndex++) {
                        boolean stop = (gameIndex == numGames - 1);

                        JSONArray playersArray = (JSONArray) config.get("Players");
                        JSONArray tribesArray = (JSONArray) config.get("Tribes");
                        if (playersArray.length() != tribesArray.length())
                            throw new Exception("Number of players must be equal to number of tribes");

                        int nPlayers = playersArray.length();
                        Run.PlayerType[] playerTypes = new Run.PlayerType[nPlayers];
                        Types.TRIBE[] tribes = new Types.TRIBE[nPlayers];

                        for (int i = 0; i < nPlayers; ++i) {
                            playerTypes[i] = Run.parsePlayerTypeStr(playersArray.getString(i));
                            tribes[i] = Run.parseTribeStr(tribesArray.getString(i));
                        }
                        Types.GAME_MODE gameMode = config.getString("Game Mode").equalsIgnoreCase("Capitals") ?
                                CAPITALS : SCORE;

                        Run.MAX_LENGTH = config.getInt("Search Depth");
                        Run.FORCE_TURN_END = config.getBoolean("Force End");
                        Run.MCTS_ROLLOUTS = config.getBoolean("Rollouts");
                        Run.POP_SIZE = config.getInt("Population Size");

                        //Portfolio and pruning variables:
                        Run.PRUNING = config.getBoolean("Pruning");
                        Run.PROGBIAS = config.getBoolean("Progressive Bias");
                        Run.K_INIT_MULT = config.getDouble("K init mult");
                        Run.T_MULT = config.getDouble("T mult");
                        Run.A_MULT = config.getDouble("A mult");
                        Run.B = config.getDouble("B");

                        JSONArray weights = null;
                        if (config.has("pMCTS Weights"))
                            weights = (JSONArray) config.get("pMCTS Weights");
                        Run.pMCTSweights = Run.getWeights(weights);

                        AGENT_SEED = config.getLong("Agents Seed");
                        GAME_SEED = config.getLong("Game Seed");
                        long levelSeed = config.getLong("Level Seed");

                        // 1. Play one game with visuals using the Level Generator:
                        if (runMode.equalsIgnoreCase("PlayLG")) {
                            System.out.println("Playing LG game: " + tribes + " " + levelSeed + " " + playerTypes + " " + gameMode + " " + stop + " " + gameIndex);
                            for (Types.TRIBE tribe : tribes) {
                                System.out.println("Playing LG game with tribe: " + tribe.getName());
                            }
                            System.out.println("Number of player types: " + playerTypes.length);
                            System.out.println("Player types: " + Arrays.toString(playerTypes));

                            play(tribes, levelSeed, playerTypes, gameMode, stop, gameIndex);

                        // 2. Play one game with visuals from a specific level file:
                        } else if (runMode.equalsIgnoreCase("PlayFromLevelFile")) {
                            String levelFile = config.getString("Level File");
                            play(levelFile, playerTypes, gameMode, stop, gameIndex);

                        // 3. Continue playing one game with visuals from a savegame
                        } else if (runMode.equalsIgnoreCase("ContinuePlayFromGameStateFile")) {
                            String saveGameFile = config.getString("Continue Game File Name");
                            load(playerTypes, saveGameFile);

                        // 4. Replay one game with visuals from specific folder
                        } else if (runMode.equalsIgnoreCase("Replay")) {
                            String replayFolder = config.getString("Replay Folder");
                            replay(playerTypes, replayFolder);
                        } else {
                            System.out.println("ERROR: run mode '" + runMode + "' not recognized.");
                        }

                        pb.step();
                    }
                }
            } else {
                System.out.println("ERROR: Couldn't find 'play.json'");
            }
        }catch(Exception e)
        {
            e.printStackTrace();
        }
    }

    private static void play(Types.TRIBE[] tribes, long levelSeed, Run.PlayerType[] playerTypes, Types.GAME_MODE gameMode, boolean stop, int gameIndex)
    {
        KeyController ki = new KeyController(true);
        ActionController ac = new ActionController();

        Game game = _prepareGame(tribes, levelSeed, playerTypes, gameMode, ac);
        Run.runGame(game, ki, ac, stop, gameIndex);
    }

    private static void play(String levelFile, Run.PlayerType[] playerTypes, Types.GAME_MODE gameMode, boolean stop, int gameIndex)
    {
        KeyController ki = new KeyController(true);
        ActionController ac = new ActionController();

        Game game = _prepareGame(levelFile, playerTypes, gameMode, ac);
        Run.runGame(game, ki, ac, stop, gameIndex);
    }


    private static void load(Run.PlayerType[] playerTypes, String saveGameFile)
    {
        KeyController ki = new KeyController(true);
        ActionController ac = new ActionController();

        long agentSeed = AGENT_SEED == -1 ? System.currentTimeMillis() + new Random().nextInt() : AGENT_SEED;

        Game game = _loadGame(playerTypes, saveGameFile, agentSeed);
        Run.runGame(game, ki, ac, true, 0);
    }


    private static void replay(Run.PlayerType[] playerTypes, String replayFolder) {
        System.out.println("Replay method called with folder: " + replayFolder);
        KeyController ki = new KeyController(true);
        ActionController ac = new ActionController();

        long agentSeed = AGENT_SEED == -1 ? System.currentTimeMillis() + new Random().nextInt() : AGENT_SEED;

        // Load the game states from the replay folder
        List<Game> gameStates = loadGameStatesFromFolder(replayFolder, playerTypes, agentSeed);

        // Run the replay using the loaded game states
        runReplay(gameStates, ki, ac);
    }

    private static List<Game> loadGameStatesFromFolder(String replayFolder, Run.PlayerType[] playerTypes, long agentSeed) {
        System.out.println("Loading game states from folder: " + replayFolder);
        List<Game> gameStates = new ArrayList<>();
        File folder = new File(replayFolder);

        // Recursively find all game.json files
        List<File> gameFiles = new ArrayList<>();
        findGameFiles(folder, gameFiles);

        if (!gameFiles.isEmpty()) {
            // Sort files based on the directory name, which is in the format "{tick}_{activeTribeID}"
            gameFiles.sort(Comparator.comparing(file -> {
                String[] parts = file.getParentFile().getName().split("_");
                int tick = Integer.parseInt(parts[0]);
                int tribeID = Integer.parseInt(parts[1]);
                return tick * 1000 + tribeID; // Ensure tick has higher sorting priority
            }));

            System.out.println("Found " + gameFiles.size() + " game state files.");
            for (File file : gameFiles) {
                System.out.println("Loading game state from file: " + file.getPath());
                Game game = _loadGame(playerTypes, file.getPath(), agentSeed);
                gameStates.add(game);
            }
        } else {
            System.out.println("No game state files found in folder.");
        }
        return gameStates;
    }
    
    private static void findGameFiles(File directory, List<File> gameFiles) {
        File[] files = directory.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.isDirectory()) {
                    findGameFiles(file, gameFiles); // Recursively search subdirectories
                } else if (file.getName().equals("game.json")) {
                    gameFiles.add(file);
                }
            }
        }
    }

    private static void runReplay(List<Game> gameStates, KeyController ki, ActionController ac) {
        System.out.println("Running replay with " + gameStates.size() + " game states.");
        GUI frame = null;
        WindowInput wi = null;
    
        for (int i = 0; i < gameStates.size(); i++) {
            Game game = gameStates.get(i);
            System.out.println("Displaying game state " + (i + 1) + " of " + gameStates.size());
    
            if (frame == null) {
                wi = new WindowInput();
                frame = new GUI(game, "Replay", ki, wi, ac, true);
                frame.addWindowListener(wi);
                frame.addKeyListener(ki);
            }
    
            // Update the GUI with the current game state
            frame.update(game.getGameState(-1), null);
    
            // Optionally add a delay between frames for better visualization
            try {
                Thread.sleep(1000); // 1 second delay
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    
        if (frame != null) {
            frame.dispose();
        }
    }

    private static Game _prepareGame(String levelFile, Run.PlayerType[] playerTypes, Types.GAME_MODE gameMode, ActionController ac)
    {
        long gameSeed = GAME_SEED == -1 ? System.currentTimeMillis() : GAME_SEED;
        if(RUN_VERBOSE) System.out.println("Game seed: " + gameSeed);

        ArrayList<Agent> players = getPlayers(playerTypes, ac);

        Game game = new Game();
        game.init(players, levelFile, gameSeed, gameMode);
        return game;
    }

    private static Game _prepareGame(Types.TRIBE[] tribes, long levelSeed, Run.PlayerType[] playerTypes, Types.GAME_MODE gameMode, ActionController ac)
    {
        long gameSeed = GAME_SEED == -1 ? System.currentTimeMillis() : GAME_SEED;

        if(RUN_VERBOSE) System.out.println("Game seed: " + gameSeed);

        ArrayList<Agent> players = getPlayers(playerTypes, ac);

        Game game = new Game();

        long levelGenSeed = levelSeed;
        if(levelGenSeed == -1)
            levelGenSeed = System.currentTimeMillis() + new Random().nextInt();

        if(RUN_VERBOSE) System.out.println("Level seed: " + levelGenSeed);

        game.init(players, levelGenSeed, tribes, gameSeed, gameMode);

        return game;
    }

    private static ArrayList<Agent> getPlayers(Run.PlayerType[] playerTypes, ActionController ac)
    {
        ArrayList<Agent> players = new ArrayList<>();
        long agentSeed = AGENT_SEED == -1 ? System.currentTimeMillis() + new Random().nextInt() : AGENT_SEED;

        if(RUN_VERBOSE)  System.out.println("Agents random seed: " + agentSeed);

        ArrayList<Integer> allIds = new ArrayList<>();
        for(int i = 0; i < playerTypes.length; ++i)
            allIds.add(i);

        for(int i = 0; i < playerTypes.length; ++i)
        {
            Agent ag = Run.getAgent(playerTypes[i], agentSeed);
            assert ag != null;
            ag.setPlayerIDs(i, allIds);
            players.add(ag);
        }
        return players;
    }

    private static Game _loadGame(Run.PlayerType[] playerTypes, String saveGameFile, long agentSeed)
    {
        ArrayList<Agent> players = new ArrayList<>();
        ArrayList<Integer> allIds = new ArrayList<>();
        for(int i = 0; i < playerTypes.length; ++i)
            allIds.add(i);

        for(int i = 0; i < playerTypes.length; ++i)
        {
            Agent ag = Run.getAgent(playerTypes[i], agentSeed);
            assert ag != null;
            ag.setPlayerIDs(i, allIds);
            players.add(ag);
        }

        Game game = new Game();
        game.init(players, saveGameFile);
        return game;
    }

}