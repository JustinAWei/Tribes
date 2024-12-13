import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.sun.net.httpserver.HttpServer;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpExchange;

import java.io.IOException;
import java.io.OutputStream;
import java.io.InputStreamReader;
import java.io.BufferedReader;
import java.net.InetSocketAddress;
import java.util.ArrayList;
import java.util.Random;

import core.Types;
import core.game.Game;
import core.game.GameState;
import org.json.JSONArray;
import org.json.JSONObject;
import players.ActionController;
import players.Agent;
import utils.GameStateSerializer;
import utils.file.IO;

public class GameServer {

    private static Game currentGame; // Class-level variable to maintain the game state

    public static void main(String[] args) throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress(8080), 0);
        server.createContext("/newGame", new NewGameHandler());
        server.createContext("/simulateAction", new SimulateActionHandler());
        server.setExecutor(null);
        server.start();
        System.out.println("Server started on port 8080");
    }

    static class NewGameHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if ("POST".equals(exchange.getRequestMethod())) {
                // Initialize a new game
                currentGame = initializeNewGame();
                
                if (currentGame != null) {
                    // Get the game state from the active player's perspective
                    GameState activePlayerState = currentGame.getGameStateForActivePlayer();

                    // Set up Gson with the custom GameStateSerializer
                    Gson gson = new GsonBuilder()
                            .registerTypeAdapter(GameState.class, new GameStateSerializer())
                            .create();

                    // Serialize the game state to JSON
                    String response = gson.toJson(activePlayerState);

                    exchange.sendResponseHeaders(200, response.getBytes().length);
                    OutputStream os = exchange.getResponseBody();
                    os.write(response.getBytes());
                    os.close();
                } else {
                    String response = "Failed to initialize new game.";
                    exchange.sendResponseHeaders(500, response.getBytes().length);
                    OutputStream os = exchange.getResponseBody();
                    os.write(response.getBytes());
                    os.close();
                }
            } else {
                exchange.sendResponseHeaders(405, -1); // Method Not Allowed
            }
        }

        // Note: This function is duplicate code from Play.java
        private Game initializeNewGame() {
            try {
                // Read the configuration from play.json
                JSONObject config = new IO().readJSON("play.json");

                if (config != null && !config.isEmpty()) {
                    // Extract player types and tribes
                    JSONArray playersArray = config.getJSONArray("Players");
                    JSONArray tribesArray = config.getJSONArray("Tribes");
                    if (playersArray.length() != tribesArray.length()) {
                        throw new Exception("Number of players must be equal to number of tribes");
                    }

                    int nPlayers = playersArray.length();
                    Run.PlayerType[] playerTypes = new Run.PlayerType[nPlayers];
                    Types.TRIBE[] tribes = new Types.TRIBE[nPlayers];

                    for (int i = 0; i < nPlayers; ++i) {
                        playerTypes[i] = Run.parsePlayerTypeStr(playersArray.getString(i));
                        tribes[i] = Run.parseTribeStr(tribesArray.getString(i));
                    }

                    // Extract game mode
                    Types.GAME_MODE gameMode = config.getString("Game Mode").equalsIgnoreCase("Capitals") ?
                            Types.GAME_MODE.CAPITALS : Types.GAME_MODE.SCORE;

                    // Extract level seed
                    long levelSeed = config.getLong("Level Seed");
                    long gameSeed = System.currentTimeMillis(); // or use a specific seed if needed

                    ActionController ac = new ActionController();
                    ArrayList<Agent> players = getPlayers(playerTypes, ac);

                    Game game = new Game();
                    game.init(players, levelSeed, tribes, gameSeed, gameMode);

                    return game;
                } else {
                    throw new Exception("ERROR: Couldn't find 'play.json'");
                }
            } catch (Exception e) {
                e.printStackTrace();
                return null;
            }
        }

        // Note: This function is duplicate code from Play.java
        private ArrayList<Agent> getPlayers(Run.PlayerType[] playerTypes, ActionController ac) {
            ArrayList<Agent> players = new ArrayList<>();
            long agentSeed = System.currentTimeMillis() + new Random().nextInt();

            ArrayList<Integer> allIds = new ArrayList<>();
            for (int i = 0; i < playerTypes.length; ++i) {
                allIds.add(i);
            }

            for (int i = 0; i < playerTypes.length; ++i) {
                Agent ag = Run.getAgent(playerTypes[i], agentSeed);
                assert ag != null;
                ag.setPlayerIDs(i, allIds);
                players.add(ag);
            }
            return players;
        }
    }

    static class SimulateActionHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if ("POST".equals(exchange.getRequestMethod())) {
                // Process the next action
                if (currentGame != null) {
                    // Example: Advance the game state
                    // You would parse the request to get the move details
                    String response = "Processed next action.";
                    exchange.sendResponseHeaders(200, response.getBytes().length);
                    OutputStream os = exchange.getResponseBody();
                    os.write(response.getBytes());
                    os.close();
                } else {
                    String response = "No game initialized.";
                    exchange.sendResponseHeaders(400, response.getBytes().length);
                    OutputStream os = exchange.getResponseBody();
                    os.write(response.getBytes());
                    os.close();
                }
            } else {
                exchange.sendResponseHeaders(405, -1); // Method Not Allowed
            }
        }
    }
}