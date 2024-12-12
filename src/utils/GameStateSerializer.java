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
        jsonObject.add("cityActions", serializeActions(gameState.getCityActions(), context));
        jsonObject.add("unitActions", serializeActions(gameState.getUnitActions(), context));
        jsonObject.add("tribeActions", context.serialize(gameState.getTribeActions()));

        // Add the serialized board to the jsonObject
        jsonObject.add("board", serializeBoard(gameState.getBoard()));

        // Add the serialized units
        jsonObject.add("units", serializeUnits(gameState.getBoard()));

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

    private <T> JsonArray convert2DArray(T[][] array, java.util.function.Function<T, Integer> converter) {
        JsonArray outerArray = new JsonArray();
        for (T[] row : array) {
            JsonArray rowArray = new JsonArray();
            for (T element : row) {
                rowArray.add(converter.apply(element));
            }
            outerArray.add(rowArray);
        }
        return outerArray;
    }

    private JsonElement serializeBoard(Board board) {
        JsonObject boardJSON = new JsonObject();

        // Helper method to convert 2D arrays
        boardJSON.add("terrains", convert2DArray(board.getTerrains(), Types.TERRAIN::getKey));
        boardJSON.add("buildings", convert2DArray(board.getBuildings(), b -> b != null ? b.getKey() : -1));
        boardJSON.add("resources", convert2DArray(board.getResources(), r -> r != null ? r.getKey() : -1));

        return boardJSON;
    }

    private JsonElement serializeUnits(Board board) {
        JsonArray unitsArray = new JsonArray();
        int[][] unitIds = board.getUnits();

        // Iterate through all positions on the board
        for (int x = 0; x < board.getSize(); x++) {
            for (int y = 0; y < board.getSize(); y++) {
                int unitId = unitIds[x][y];
                // Only process positions where there's actually a unit (unitId != 0)
                if (unitId != 0) {
                    Unit unit = board.getUnitAt(x, y);
                    if (unit != null) {  // Double check unit exists
                        JsonObject unitJson = new JsonObject();
                        // Basic unit info
                        unitJson.addProperty("id", unitId);
                        unitJson.addProperty("x", x);
                        unitJson.addProperty("y", y);
                        unitJson.addProperty("tribeId", unit.getTribeId());
                        unitJson.addProperty("cityId", unit.getCityId());
                        unitJson.addProperty("type", unit.getType().toString());
                        unitJson.addProperty("currentHP", unit.getCurrentHP());
                        unitJson.addProperty("maxHP", unit.getMaxHP());
                        unitJson.addProperty("kills", unit.getKills());
                        unitJson.addProperty("isVeteran", unit.isVeteran());

                        // For water units, include base land unit type
                        if (unit.getType().isWaterUnit()) {
                            Types.UNIT baseLandUnit = board.getBaseLandUnit(unit);
                            unitJson.addProperty("baseLandUnit", baseLandUnit.toString());
                        }

                        unitsArray.add(unitJson);
                    }
                }
            }
        }
        return unitsArray;
    }
}