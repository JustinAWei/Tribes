package utils;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonSerializationContext;
import com.google.gson.JsonSerializer;
import core.Types;
import core.game.Board;
import core.game.GameState;
import core.actions.Action;
import org.json.JSONArray;

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

        // Add the serialized board to the jsonObject
        jsonObject.add("board", serializeBoard(gameState.getBoard()));

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

    private JsonElement serializeBoard(Board board) {
        JsonObject boardJSON = new JsonObject();

        // Serialize Terrains
        Types.TERRAIN[][] terrains = board.getTerrains();
        JsonArray terrainsArray = new JsonArray();
        for (Types.TERRAIN[] row : terrains) {
            JsonArray rowArray = new JsonArray();
            for (Types.TERRAIN terrain : row) {
                rowArray.add(terrain.getKey());
            }
            terrainsArray.add(rowArray);
        }
        boardJSON.add("terrains", terrainsArray);

        // Serialize Buildings
        Types.BUILDING[][] buildings = board.getBuildings();
        JsonArray buildingsArray = new JsonArray();
        for (Types.BUILDING[] row : buildings) {
            JsonArray rowArray = new JsonArray();
            for (Types.BUILDING building : row) {
                rowArray.add(building != null ? building.getKey() : -1);
            }
            buildingsArray.add(rowArray);
        }
        boardJSON.add("buildings", buildingsArray);

        // Serialize Resources
        Types.RESOURCE[][] resources = board.getResources();
        JsonArray resourcesArray = new JsonArray();
        for (Types.RESOURCE[] row : resources) {
            JsonArray rowArray = new JsonArray();
            for (Types.RESOURCE resource : row) {
                rowArray.add(resource != null ? resource.getKey() : -1);
            }
            resourcesArray.add(rowArray);
        }
        boardJSON.add("resources", resourcesArray);

        return boardJSON;
    }
}