package utils;

import java.net.HttpURLConnection;
import java.net.URL;
import java.io.OutputStream;
import java.io.BufferedReader;
import java.io.InputStreamReader;

public class PostRequestSender {

    public static String sendPostRequest(String targetUrl, String jsonInputString) {
        StringBuilder response = new StringBuilder();
        try {
            // Define the target URL
            URL url = new URL(targetUrl);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();

            // Set request properties
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json; utf-8");
            conn.setRequestProperty("Accept", "application/json");
            conn.setDoOutput(true);

            // Send request
            try (OutputStream os = conn.getOutputStream()) {
                byte[] input = jsonInputString.getBytes("utf-8");
                os.write(input, 0, input.length);
            }

            // Read response
            try (BufferedReader br = new BufferedReader(
                    new InputStreamReader(conn.getInputStream(), "utf-8"))) {
                String responseLine;
                while ((responseLine = br.readLine()) != null) {
                    response.append(responseLine.trim());
                }
            }
            conn.disconnect();
        } catch (Exception e) {
            e.printStackTrace();
            return "Error: " + e.getMessage();
        }
        return response.toString();
    }

    public static void main(String[] args) {
        String url = "http://localhost:8000/receive";
        String jsonPayload = "{\"key\":\"value\"}";
        String response = sendPostRequest(url, jsonPayload);
        System.out.println("Response: " + response);
    }
}
