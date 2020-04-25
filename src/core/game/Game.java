package core.game;

import core.TribesConfig;
import core.Types;
import core.actions.Action;
import core.actions.unitactions.Recover;
import core.actions.unitactions.factory.RecoverFactory;
import core.actors.Building;
import core.actors.City;
import core.actors.Temple;
import core.actors.Tribe;
import core.actors.units.Unit;
import players.Agent;
import players.HumanAgent;
import utils.ElapsedCpuTimer;
import utils.GUI;
import utils.Vector2d;
import utils.WindowInput;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Random;

import static core.Constants.*;

public class Game {

    private boolean FORCE_FULL_OBSERVABILITY = false;

    // State of the game (objects, ticks, etc).
    private GameState gs;

    // GameState objects for players to make decisions
    private GameState[] gameStateObservations;

    // Seed for the game state.
    private long seed;

    //Random number generator for the game.
    private Random rnd;

    // List of players of the game
    private Agent[] players;

    //Number of players of the game.
    private int numPlayers;

    /**
     * Constructor of the game
     */
    public Game()
    {}

    /**
     * Initializes the game. This method does the following:
     *   Sets the players of the game, the number of players and their IDs
     *   Initializes the array to hold the player game states.
     *   Assigns the tribes that will play the game.
     *   Creates the board according to the above information and resets the game so it's ready to start.
     *   Turn order: by default, turns run following the order in the tribes array.
     * @param players Players of the game.
     * @param tribes Tribes to play the game with. Players and tribes related by position in array lists.
     * @param filename Name of the file with the level information.
     * @param seed Seed for the game (used only for board generation)
     * @param gameMode Game Mode for this game.
     */
    public void init(ArrayList<Agent> players, ArrayList<Tribe> tribes, String filename, long seed, Types.GAME_MODE gameMode) {

        //Initiate the bare bones of the main game classes
        this.seed = seed;
        this.rnd = new Random(seed);
        this.gs = new GameState(rnd, gameMode);

        if(players.size() != tribes.size())
        {
            System.out.println("ERROR: Number of tribes must equal the number of players.");
        }

        Tribe[] tribesArray = new Tribe[tribes.size()];
        for (int i = 0; i < tribesArray.length; ++i)
        {
            tribesArray[i] = tribes.get(i);
        }

        //Create the players and agents to control them
        numPlayers = players.size();
        this.players = new Agent[numPlayers];
        for(int i = 0; i < numPlayers; ++i)
        {
            this.players[i] = players.get(i);
            this.players[i].setPlayerID(i);
        }
        this.gameStateObservations = new GameState[numPlayers];

        //Assign the tribes to the players
//        this.gs.assignTribes(tribes);

        this.gs.init(filename, tribesArray);

        updateAssignedGameStates();
    }

    public void init(ArrayList<Agent> players, String fileName){

        GameLoader gameLoader = new GameLoader(fileName);
        this.seed = gameLoader.getSeed();
        this.rnd = new Random(seed);
        Tribe[] tribes = gameLoader.getTribes();

        this.gs = new GameState(rnd, gameLoader.getGame_mode(), tribes, gameLoader.getBoard(), gameLoader.getTick());
        this.gs.setGameIsOver(gameLoader.getGameIsOver());

        if(players.size() != tribes.length)
        {
            System.out.println("ERROR: Number of tribes must equal the number of players. The file only has " + tribes.length + " players.");
        }

        //Create the players and agents to control them
        numPlayers = players.size();
        this.players = new Agent[numPlayers];
        for(int i = 0; i < numPlayers; ++i)
        {
            this.players[i] = players.get(i);
            this.players[i].setPlayerID(i);
        }

        this.gameStateObservations = new GameState[numPlayers];

        updateAssignedGameStates();

    }


//    /**
//     * Resets the game, providing a seed.
//     * @param repeatLevel true if the same level should be played.
//     * @param filename Name of the file with the level information.
//     */
//    public void reset(boolean repeatLevel, String filename)
//    {
//        this.seed = repeatLevel ? seed : System.currentTimeMillis();
//        resetGame(filename, numPlayers);
//    }
//
//    /**
//     * Resets the game, providing a seed.
//     * @param seed new seed for the game.
//     * @param filename Name of the file with the level information.
//     */
//    public void reset(int seed, String filename)
//    {
//        this.seed = seed;
//        resetGame(filename, numPlayers);
//    }

//    /**
//     * Resets the game, creating the original game state (and level) and assigning the initial
//     * game state views that each player will have.
//     * @param filename Name of the file with the level information.
//     */
//    private void resetGame(String filename)
//    {
//        this.gs.init(filename);
//        updateAssignedGameStates();
//    }




    /**
     * Runs a game once. Receives frame and window input. If any is null, forces a run with no visuals.
     * @param frame window to draw the game
     * @param wi input for the window.
     */
    public void run(GUI frame, WindowInput wi)
    {
        if (frame == null || wi == null)
            VISUALS = false;
        boolean firstEnd = true;

        while(!gameOver()) {
            // Loop while window is still open, even if the game ended.
            // If not playing with visuals, loop while the game's not ended.
            tick(frame);

            // Check end of game
            if (firstEnd && gameOver()) {
                firstEnd = false;

                if (!VISUALS) {
                    // The game has ended, end the loop if we're running without visuals.
                    break;
                }
            }
        }

        terminate();
    }

    /**
     * Ticks the game forward. Asks agents for actions and applies returned actions to obtain the next game state.
     */
    private void tick (GUI frame) {
        if (VERBOSE) {
            //System.out.println("tick: " + gs.getTick());
        }

        Tribe[] tribes = gs.getTribes();
        for (int i = 0; i < numPlayers; i++) {
            Tribe tribe = tribes[i];

            if(tribe.getWinner() == Types.RESULT.LOSS)
                continue; //We don't do anything for tribes that have already lost.


            //play the full turn for this player
            processTurn(i, tribe, frame);

            // Save Game
            GameSaver.writeTurnFile(gs, getBoard(), seed);

            //GameLoader gl = new GameLoader("save/" + this.seed + "/"+ gs.getTick() + "_" + gs.getActiveTribeID() +"/game.json");

            //it may be that this player won the game, no more playing.
            if(gameOver())
            {
                return;
            }
        }

        //All turns passed, time to increase the tick.
        gs.incTick();
    }

    /**
     * Process a turn for a given player. It queries the player for an action until no more
     * actions are available or the player returns a EndTurnAction action.
     * @param playerID ID of the player whose turn is being processed.
     * @param tribe tribe that corresponds to this player.
     */
    private void processTurn(int playerID, Tribe tribe, GUI frame)
    {
        //Init the turn for this tribe (stars, unit reset, etc).
        gs.initTurn(tribe);

        //Compute the initial player actions and assign the game states.
        gs.computePlayerActions(tribe);
        updateAssignedGameStates();

        //Take the player for this turn
        Agent ag = players[playerID];

        //start the timer to the max duration
        ElapsedCpuTimer ect = new ElapsedCpuTimer();
        ect.setMaxTimeMillis(TURN_TIME_MILLIS);
        boolean continueTurn = true;
        int curActionCounter = 0;

        while(continueTurn)
        {
            //get one action from the player
            Action action = ag.act(gameStateObservations[playerID], ect);

//            System.out.println(gs.getTick() + " " + curActionCounter + " " + action + "; stars: " + gs.getBoard().getTribe(playerID).getStars());
            curActionCounter++;

            //note down the remaining time to use it for the next iteration
            long remaining = ect.remainingTimeMillis();

            //play the action in the game and update the available actions list
            gs.next(action);
            gs.computePlayerActions(tribe);

            updateAssignedGameStates();

            // Update GUI after every action
            // Paint game state
            if (VISUALS && frame != null) {
                if(FORCE_FULL_OBSERVABILITY)
                    frame.update(getGameState(-1));
                else
                    frame.update(gameStateObservations[gs.getActiveTribeID()]);        //Partial Obs
            }

            //the timer needs to be updated to the remaining time, not counting action computation.
            ect.setMaxTimeMillis(remaining);

            //Continue this turn if there are still available actions. If the agent is human, let him play for now.
            continueTurn = !gs.isTurnEnding();
            if(!(ag instanceof HumanAgent))
                continueTurn &= gs.existAvailableActions(tribe) && !ect.exceededMaxTime();
        }

        //Ends the turn for this tribe (units that didn't move heal).
        gs.endTurn(tribe);
    }




    /**
     * This method call all agents' end-of-game method for post-processing.
     * Agents receive their final game state and reward
     */
    @SuppressWarnings("UnusedReturnValue")
    private void terminate() {

        Tribe[] tribes = gs.getTribes();
        for (int i = 0; i < numPlayers; i++) {
            Agent ag = players[i];
            ag.result(gs.copy(), tribes[i].getScore());
        }
    }

    /**
     * Returns the winning status of all players.
     * @return the winning status of all players.
     */
    public Types.RESULT[] getWinnerStatus()
    {
        //Build the results array
        Tribe[] tribes = gs.getTribes();
        Types.RESULT[] results = new Types.RESULT[numPlayers];
        for (int i = 0; i < numPlayers; i++) {
            Tribe tribe = tribes[i];
            results[i] = tribe.getWinner();
        }
        return results;
    }

    /**
     * Returns the current scores of all players.
     * @return the current scores of all players.
     */
    public int[] getScores()
    {
        //Build the results array
        Tribe[] tribes = gs.getTribes();
        int[] scores = new int[numPlayers];
        for (int i = 0; i < numPlayers; i++) {
            scores[i] = tribes[i].getScore();
        }
        return scores;
    }

    /**
     * Updates the state observations for all players with copies of the
     * current game state, adapted for PO.
     */
    private void updateAssignedGameStates() {

        //TODO: Probably we don't need to do this for all players, just the active one.
        for (int i = 0; i < numPlayers; i++) {
            gameStateObservations[i] = getGameState(i);
        }
    }

    /**
     * Returns the game state as seen for the player with the index playerIdx. This game state
     * includes only the observations that are visible if partial observability is enabled.
     * @param playerIdx index of the player for which the game state is generated.
     * @return the game state.
     */
    private GameState getGameState(int playerIdx) {
        return gs.copy(playerIdx);
    }

    /**
     * Returns the game board.
     * @return the game board.
     */
    public Board getBoard()
    {
        return gs.getBoard();
    }

    public Agent[] getPlayers() {
        return players;
    }

    /**
     * Method to identify the end of the game. If the game is over, the winner is decided.
     * The winner of a game is determined by TribesConfig.GAME_MODE and TribesConfig.MAX_TURNS
     * @return true if the game has ended, false otherwise.
     */
    boolean gameOver() {
        return gs.gameOver();
    }

}
