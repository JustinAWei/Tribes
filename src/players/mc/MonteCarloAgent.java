package players.mc;

import com.google.gson.*;
import core.Types;
import core.actions.Action;
import core.actions.cityactions.CityAction;
import core.actions.tribeactions.EndTurn;
import core.actions.unitactions.UnitAction;
import core.actors.Tribe;
import core.game.GameState;
import org.json.JSONArray;
import org.json.JSONObject;
import players.Agent;
import players.heuristics.StateHeuristic;
import utils.ElapsedCpuTimer;
import utils.PostRequestSender;
import utils.stats.StatSummary;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import utils.GameStateSerializer;

public class MonteCarloAgent extends Agent {

    private Random m_rnd;
    private MCParams params;
    private StateHeuristic heuristic;
    private int lastTurn;
    private int actionTurnCounter;
    private int fmCalls;

    public MonteCarloAgent(long seed, MCParams params)
    {
        super(seed);
        m_rnd = new Random(seed);
        this.params = params;
        this.lastTurn = -1;
        this.actionTurnCounter = 0;
    }

    @Override
    public Action act(GameState gs, ElapsedCpuTimer ect) {
        PostRequestSender postRequestSender = new PostRequestSender();
        String url = "http://localhost:8000/receive";
        JsonObject jsonObject = new JsonObject();

        // Serialize the GameState
        Gson gson = new GsonBuilder()
                .registerTypeAdapter(GameState.class, new GameStateSerializer())
                .registerTypeAdapter(Types.TRIBE.class, (JsonSerializer<Types.TRIBE>) (src, typeOfSrc, context) ->
                        new JsonPrimitive(src.toString()))
                .registerTypeAdapter(Types.RESULT.class, (JsonSerializer<Types.RESULT>) (src, typeOfSrc, context) ->
                        new JsonPrimitive(src.toString()))
                .create();
        JsonElement gameStateJson = gson.toJsonTree(gs);;
        jsonObject.add("gameState", gameStateJson); // Add serialized game state

        String response = postRequestSender.sendPostRequest(url, jsonObject.toString());
        System.out.println(response);
        JSONObject jsonResponse = new JSONObject(response);
        System.out.println(jsonResponse);

        // turn into array
        JSONArray jsonArray = (JSONArray) jsonResponse.get("action");
        System.out.println(jsonArray);

        // Turn the tensor into an action item
        Integer category = (Integer) jsonArray.get(0);
        // first, switch on Tribe, City, Unit action

        // then, get the id
        Integer id = (Integer) jsonArray.get(1);

        // then, get the action
        Integer actionId = (Integer) jsonArray.get(2);
        Types.ACTION actionType = Types.ACTION.values()[actionId];

        // then, get the right param based on x,y
        Integer x = (Integer) jsonArray.get(3);
        Integer y = (Integer) jsonArray.get(4);

        ArrayList<Action> allActions = gs.getAllAvailableActions();

        // filter by action type
        // Filter actions by type and coordinates
        ArrayList<Action> filteredActions = new ArrayList<>();

        for (Action a : allActions) {
            if (a.getActionType() == actionType) {
                if (category == 1 && a instanceof CityAction && id == ((CityAction) a).getCityId()) {
                    filteredActions.add(a);
                } else if (category == 2 && a instanceof UnitAction && id == ((UnitAction) a).getUnitId()) {
                    filteredActions.add(a);
                } else if (category == 0) {
                    filteredActions.add(a);
                }
            }
        }

        System.out.println(filteredActions);

        // TODO: now we match the params x,y

        return filteredActions.get(0);
    }


    /**
     * Executes a Monte Carlo rollout.
     * @param gs current game state (a copy)
     * @param act action to start the rollout with.
     * @return the score of the state found at the end of the rollout, as evaluated by a heuristic
     */
    private double rollout(GameState gs, Action act)
    {
        GameState gsCopy = copyGameState(gs);
        boolean end = false;
        int step = 0;
        int turnEndCountDown = params.FORCE_TURN_END; // We force an EndTurn action every FORCE_TURN_END actions in the rollout.
        boolean run;

        while(!end)
        {
            run = true;

            //If it's time to force a turn end, do it
            if(turnEndCountDown == 0)
            {
                EndTurn endTurn = new EndTurn(gsCopy.getActiveTribeID());
                boolean canEndTurn = endTurn.isFeasible(gsCopy);

                if(canEndTurn) //check if we can actually end the turn (game may be expecting a non EndTurn action in Tribes).
                {
                    advance(gsCopy, endTurn, true);
                    turnEndCountDown = params.FORCE_TURN_END;
                    run = false;
                }
            }

            //Actually run the action
            if(run)
            {
                advance(gsCopy, act, true);
                turnEndCountDown--;
            }

            //Check if it's time to end this rollout. 1) either because it's a game end, 2) we've reached the end of it...
            step++;
            end = gsCopy.isGameOver() || (step == params.ROLLOUT_LENGTH);

            // ... or 3) we have no more thinking time available (agent's budget)
            boolean budgetOver = (params.stop_type == params.STOP_FMCALLS && fmCalls >= params.num_fmcalls);
            end |= budgetOver;

            if(!end)
            {
                //If we can continue, pick another action to run at random
                ArrayList<Action> allActions = gsCopy.getAllAvailableActions();
                int numActions = allActions.size();
                if(numActions == 1) {
                    //If there's only 1 action available, it should be an EndTurn
                    act = allActions.get(0);
                    if(act.getActionType() == Types.ACTION.END_TURN)
                        turnEndCountDown = params.FORCE_TURN_END + 1;
                    else
                        System.out.println("Warning: Unexpected non-EndTurn action in MC player");

                }else
                {
                    //If there are many actions, we select the next action for the rollout at random, avoiding EndTurn.
                    do {
                        int actIdx = m_rnd.nextInt(numActions);
                        act = allActions.get(actIdx);

                    }  while(act.getActionType() == Types.ACTION.END_TURN);
                }
            }
        }

        //We evaluate the state found at the end of the rollout with an heuristic.
        return heuristic.evaluateState(gs, gsCopy);
    }

    /**
     * Wrapper for advancing the game state. Updates the count of Forward Model usages.
     * @param gs game state to advance
     * @param act action to advance it with.
     * @param computeActions true if the game state should compute the available actions after advancing the state.
     */
    private void advance(GameState gs, Action act, boolean computeActions)
    {
        gs.advance(act, computeActions);
        fmCalls++;
    }

    /**
     * The technology trees of the opponents are always empty (no technology is researched).
     * As a simple case of gamestate injection, we research N technologies (N=turn/2) for them
     * @param gs current game state.
     */
    private void initTribesResearch(GameState gs)
    {
        int turn = gs.getTick();
        int techsToResearch = (int) (turn / 2.0);
        for(Tribe t : gs.getTribes())
        {
            if(t.getTribeId() != this.playerID)
            {
                for(int i = 0; i < techsToResearch; ++i)
                    t.getTechTree().researchAtRandom(this.m_rnd);
            }
        }
    }

    public GameState copyGameState(GameState gs)
    {
        GameState gsCopy = gs.copy();
        initTribesResearch(gsCopy);
        return gsCopy;
    }

    @Override
    public Agent copy() {
        return null; //not needed.
    }
}
