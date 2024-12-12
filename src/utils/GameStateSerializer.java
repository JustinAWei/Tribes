package utils;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonSerializationContext;
import com.google.gson.JsonSerializer;
import core.Types;
import core.actors.units.Unit;
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
        jsonObject.add("cityActions", context.serialize(gameState.getCityActions()));
        jsonObject.add("unitActions", context.serialize(gameState.getUnitActions()));
        jsonObject.add("tribeActions", context.serialize(gameState.getTribeActions()));
        jsonObject.add("board", context.serialize(gameState.getBoard()));
        jsonObject.add("tribes", context.serialize(gameState.getTribes()));

        // Serialize ranking
        jsonObject.add("ranking", context.serialize(gameState.getCurrentRanking()));

        return jsonObject;
    }
}