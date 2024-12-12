package players.mcts;

import com.google.gson.*;
import core.Types;
import core.actions.Action;
import core.actions.tribeactions.EndTurn;
import core.game.Game;
import core.game.GameState;
import org.json.JSONObject;
import players.Agent;
import utils.ElapsedCpuTimer;
import utils.GameStateSerializer;
import utils.PostRequestSender;

import java.util.ArrayList;
import java.util.Random;

import static core.Constants.TURN_TIME_MILLIS;

public class MCTSPlayer extends Agent {

    private Random m_rnd;
    private MCTSParams params;

    public MCTSPlayer(long seed)
    {
        super(seed);
        m_rnd = new Random(seed);
        this.params = new MCTSParams();
    }

    public MCTSPlayer(long seed, MCTSParams params) {
        this(seed);
        this.params = params;
    }

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

        //Gather all available actions:
        ArrayList<Action> allActions = gs.getAllAvailableActions();

        if(allActions.size() == 1)
            return allActions.get(0); //EndTurn, it's possible.

        ArrayList<Action> rootActions = params.PRIORITIZE_ROOT ? determineActionGroup(gs, m_rnd) : allActions;
        if(rootActions == null)
            return new EndTurn();

        SingleTreeNode m_root = new SingleTreeNode(params, m_rnd, rootActions.size(), rootActions, this.playerID);
        m_root.setRootGameState(m_root, gs, allPlayerIDs);

        m_root.mctsSearch(ect);

        return rootActions.get(m_root.mostVisitedAction());

    }


    @Override
    public Agent copy() {
        return null;
    }

}