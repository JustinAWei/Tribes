package utils;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonSerializationContext;
import com.google.gson.JsonSerializer;
import core.game.GameState;
import core.actions.Action;

import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class GameStateSerializer implements JsonSerializer<GameState> {

    @Override
    public JsonElement serialize(GameState gameState, Type typeOfSrc, JsonSerializationContext context) {
        JsonObject jsonObject = new JsonObject();

        // Serialize basic fields
        jsonObject.addProperty("gameMode", gameState.getGameMode().toString());
        jsonObject.addProperty("tick", gameState.getTick());
//        jsonObject.addProperty("turnMustEnd", gameState.isTurnEnding());
        jsonObject.addProperty("gameIsOver", gameState.isGameOver());

        // Serialize actions
        jsonObject.add("cityActions", serializeActions(gameState.getCityActions(), context));
        jsonObject.add("unitActions", serializeActions(gameState.getUnitActions(), context));
        jsonObject.add("tribeActions", context.serialize(gameState.getTribeActions()));

        // Serialize ranking
        jsonObject.add("ranking", context.serialize(gameState.getCurrentRanking()));

        return jsonObject;
    }

    private JsonElement serializeActions(HashMap<Integer, ArrayList<Action>> actionsMap, JsonSerializationContext context) {
        JsonObject actionsJson = new JsonObject();
        for (Map.Entry<Integer, ArrayList<Action>> entry : actionsMap.entrySet()) {
            actionsJson.add(entry.getKey().toString(), context.serialize(entry.getValue()));
        }
        return actionsJson;
    }
}