package players.ppo;

import com.google.gson.*;
import core.Types;
import core.actions.Action;
import core.actions.cityactions.*;
import core.actions.tribeactions.BuildRoad;
import core.actions.tribeactions.ResearchTech;
import core.actions.unitactions.Attack;
import core.actions.unitactions.Move;
import core.actions.unitactions.UnitAction;
import core.actors.Actor;
import core.actors.units.Unit;
import core.game.Board;
import core.game.GameState;
import org.json.JSONArray;
import org.json.JSONObject;
import players.Agent;
import players.mcts.MCTSParams;
import utils.ElapsedCpuTimer;
import utils.GameStateSerializer;
import utils.PostRequestSender;
import utils.Vector2d;

import java.util.ArrayList;
import java.util.Random;

public class PPOAgent extends Agent {

    public PPOAgent(long seed)
    {
        super(seed);
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
        JSONObject jsonResponse = new JSONObject(response);
        System.out.println("Parsing response!");
        System.out.println(jsonResponse);

        // turn into array
        JSONArray jsonArray = (JSONArray) jsonResponse.get("action");

        // Turn the tensor into an action item

        // then, get the right param based on x,y
        Integer x1 = (Integer) jsonArray.get(0);
        Integer y1 = (Integer) jsonArray.get(1);

        // then, get the right param based on x,y
        Integer x2 = (Integer) jsonArray.get(2);
        Integer y2 = (Integer) jsonArray.get(3);

        Integer actionId = (Integer) jsonArray.get(4);
        Types.ACTION actionType = Types.ACTION.values()[actionId];
        System.out.println("Action type: " + actionType);

        int typeInfo = (Integer) jsonArray.get(5);

        Board board = gs.getBoard();
        ArrayList<Action> allActions = gs.getAllAvailableActions();

        // filter by action type
        // Filter actions by type and coordinates
        ArrayList<Action> filteredActions = new ArrayList<>();

        for (Action a : allActions) {
            System.out.println("Checking action: " + a);
            System.out.println("Action type from action: " + a.getActionType());
            System.out.println("Action type we're looking for: " + actionType);
            System.out.println("Are they equal? " + (a.getActionType() == actionType));

            if (a.getActionType() != actionType) {
                continue;
            }
            if (a instanceof CityAction) {
                int cityId = board.getCityIdAt(x1,y1);
                if (cityId == ((CityAction) a).getCityId()) {
                    // Build
                    if (actionType == Types.ACTION.BUILD) {
                        Types.BUILDING buildingType = ((Build) a).getBuildingType();
                        Types.BUILDING buildingTypeInfo = Types.BUILDING.getTypeByKey(typeInfo);
                        Vector2d targetPos = ((Build) a).getTargetPos();
                        boolean targetPosMatches = targetPos.x == x2
                                && targetPos.y == y2;
                        if (buildingType == buildingTypeInfo && targetPosMatches) {
                            filteredActions.add(a);
                        }
                    }
                    // Spawn
                    else if (actionType == Types.ACTION.SPAWN) {
                        Types.UNIT unitType = ((Spawn) a).getUnitType();
                        Types.UNIT unitTypeInfo = Types.UNIT.getTypeByKey(typeInfo);
                        if (unitType == unitTypeInfo) {
                            filteredActions.add(a);
                            System.out.println("Equal: " + unitType + " " + unitTypeInfo);
                        } else {
                            System.out.println("Not equal: " + unitType + " " + unitTypeInfo);
                        }
                    }
                    else if (actionType == Types.ACTION.RESOURCE_GATHERING) {
                        // check targetPos matches x2,y2
                        ResourceGathering resourceGathering = (ResourceGathering) a;
                        boolean targetPosMatches = resourceGathering.getTargetPos().x == x2
                                && resourceGathering.getTargetPos().y == y2;
                        System.out.println("Target pos: " + resourceGathering.getTargetPos().x + ", " + resourceGathering.getTargetPos().y);
                        if (targetPosMatches) {
                            filteredActions.add(a);
                        }
                    }
                    else if (actionType == Types.ACTION.LEVEL_UP) {
//                        System.out.println("Level up");
                        // make sure it matches the bonus type
                        LevelUp levelUp = (LevelUp) a;
                        int bonusType = levelUp.getBonus().ordinal();
//                        System.out.println("Bonus Type: " + bonusType);
//                        System.out.println("Bonus Level: " + typeInfo);
                        if (bonusType == typeInfo) {
                            filteredActions.add(a);
                        }
                    }
                }
            } else if (a instanceof UnitAction) {
                try {
                    Unit unit = board.getUnitAt(x1, y1);
                    int unitId = unit.getActorId();
                    if (unitId == ((UnitAction) a).getUnitId()) {
                        if (actionType == Types.ACTION.ATTACK) {
                            // targetId needs to match x2,y2
                            Attack attack = (Attack) a;
                            int targetId = attack.getTargetId();
                            Actor actor = board.getActor(targetId);
                            boolean targetMatches = actor.getPosition().x == x2 && actor.getPosition().y == y2;
                            if (targetMatches) {
                                filteredActions.add(a);
                            }
                        }
                        else if (actionType == Types.ACTION.MOVE) {
                            // destination needs to match x2,y2
                            Move move = (Move) a;
                            boolean destinationMatches = move.getDestination().x == x2
                                    && move.getDestination().y == y2;

                            if (destinationMatches) {
                                filteredActions.add(a);
                            }
                        }
                        else if (actionType == Types.ACTION.CAPTURE || actionType == Types.ACTION.RECOVER || actionType == Types.ACTION.EXAMINE || actionType == Types.ACTION.MAKE_VETERAN || actionType == Types.ACTION.DISBAND || actionType == Types.ACTION.UPGRADE_BOAT || actionType == Types.ACTION.UPGRADE_SHIP) {
                            filteredActions.add(a);
                        }
                    }
                } catch (Exception e) {
                    System.out.println("Error occurred with action: " + a);
                    System.out.println("Coordinates: x1 = " + x1 + ", y1 = " + y1);
                    System.out.println("Board details:");
                    System.out.println("Active Tribe ID: " + board.getActiveTribeID());
                    System.out.println("Terrain at " + x1 + "," + y1 + ": " + board.getTerrainAt(x1, y1));
                    System.out.println("Resource at " + x1 + "," + y1 + ": " + board.getResourceAt(x1, y1)); 
                    System.out.println("Building at " + x1 + "," + y1 + ": " + board.getBuildingAt(x1, y1));
                    System.out.println("City ID at " + x1 + "," + y1 + ": " + board.getCityIdAt(x1, y1));
                    System.out.println("Unit ID at " + x1 + "," + y1 + ": " + board.getUnitIDAt(x1, y1));
                    System.out.println("Unit array at " + x1 + "," + y1 + ": " + board.getUnits()[x1][y1]);
                    System.out.println("Full units array:");
                    int[][] units = board.getUnits();
                    for(int i = 0; i < units.length; i++) {
                        for(int j = 0; j < units[i].length; j++) {
                            if(units[i][j] != 0) {
                                System.out.println("Unit at " + i + "," + j + ": " + units[i][j]);
                            }
                        }
                    }
                    e.printStackTrace();
                }
            }
            // Research
            // check action type
            else if (actionType == Types.ACTION.RESEARCH_TECH) {
                Types.TECHNOLOGY techType = ((ResearchTech) a).getTech();
                int techTypeInt = techType.ordinal();
//                System.out.println("Tech Type: " + techTypeInt);
//                System.out.println("Tech Tech Type: " + typeInfo);
                if (techTypeInt == typeInfo) {
                    filteredActions.add(a);
                }
            }
            else if (actionType == Types.ACTION.BUILD_ROAD) {
                // check position
                BuildRoad buildRoad = (BuildRoad) a;
                boolean positionMatches = buildRoad.getPosition().x == x1 && buildRoad.getPosition().y == y1;
                if (positionMatches) {
                    filteredActions.add(a);
                }
            }
            else if (actionType == Types.ACTION.END_TURN) {
                filteredActions.add(a);
            }
        }

        System.out.println("Filtered actions: " + filteredActions);
        System.out.println("All actions: " + allActions);

        return filteredActions.getFirst();
    }

    public Agent copy() {
        return null;
    }
}