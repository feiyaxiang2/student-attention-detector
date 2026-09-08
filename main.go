package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"

	"github.com/joho/godotenv"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

type AIAgentEvent struct {
	UserID string `json:"user_id"`
	Status string `json:"status"`
}

func aiAgentHandler(w http.ResponseWriter, r *http.Request) {
	enableCORS(w)

	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var event AIAgentEvent
	err := json.NewDecoder(r.Body).Decode(&event)
	if err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if event.UserID == "" || event.Status == "" {
		http.Error(w, "Missing user_id or status", http.StatusBadRequest)
		return
	}

	log.Printf("[AI AGENT] User: %s | Status: %s", event.UserID, event.Status)
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "received"})
}

func startCalibrationHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.URL.Query().Get("user_id")
	if userID == "" {
		http.Error(w, "user_id required", http.StatusBadRequest)
		return
	}

	// Run with --calibration_only flag
	cmd := exec.Command("python", "run_demo_test.py", "--user_id="+userID, "--calibration_only=True")
	err := cmd.Start()
	if err != nil {
		log.Printf("Failed to start calibration for user %s: %v", userID, err)
		http.Error(w, "Failed to start calibration process", http.StatusInternalServerError)
		return
	}

	log.Printf("Started calibration-only session for user %s with PID %d", userID, cmd.Process.Pid)
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{
		"status":  "calibration-only-started",
		"user_id": userID,
		"pid":     fmt.Sprint(cmd.Process.Pid),
	})
}

func startSessionHandler(w http.ResponseWriter, r *http.Request) {
	userID := r.URL.Query().Get("user_id")
	if userID == "" {
		http.Error(w, "user_id required", http.StatusBadRequest)
		return
	}

	// Start the Python subprocess
	cmd := exec.Command("python", "run_demo_test.py", "--user_id="+userID)
	err := cmd.Start()
	if err != nil {
		log.Printf("Failed to start process for user %s: %v", userID, err)
		http.Error(w, "Failed to start Python process", http.StatusInternalServerError)
		return
	}

	// Optionally: save PID and userID to a map
	log.Printf("Started session for user %s with PID %d", userID, cmd.Process.Pid)
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{
		"status":  "started",
		"user_id": userID,
		"pid":     fmt.Sprint(cmd.Process.Pid),
	})
}

type CalibrationRecord struct {
	DeviceID   string      `json:"device_id" bson:"device_id"`
	PitchRange interface{} `json:"pitch_range" bson:"pitch_range"`
	YawRange   interface{} `json:"yaw_range" bson:"yaw_range"`
}

func calibrationHandler(coll *mongo.Collection) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		enableCORS(w)

		switch r.Method {
		case "GET":
			deviceID := r.URL.Query().Get("device_id")
			if deviceID == "" {
				http.Error(w, "Missing device_id parameter", http.StatusBadRequest)
				return
			}
			var result CalibrationRecord
			err := coll.FindOne(context.TODO(), bson.M{"device_id": deviceID}).Decode(&result)
			if err != nil {
				http.Error(w, "No calibration data found", http.StatusNotFound)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(result)

		case "POST":
			var data CalibrationRecord
			err := json.NewDecoder(r.Body).Decode(&data)
			if err != nil {
				http.Error(w, "Invalid JSON", http.StatusBadRequest)
				return
			}
			if data.DeviceID == "" || data.PitchRange == nil || data.YawRange == nil {
				http.Error(w, "Missing required fields", http.StatusBadRequest)
				return
			}
			filter := bson.M{"device_id": data.DeviceID}
			update := bson.M{"$set": data}
			opts := options.Update().SetUpsert(true)
			_, err = coll.UpdateOne(context.TODO(), filter, update, opts)
			if err != nil {
				http.Error(w, "Database error", http.StatusInternalServerError)
				return
			}
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(map[string]string{"status": "success"})
			log.Printf("Saving calibration for device_id=%s", data.DeviceID)
			log.Printf("  pitch_range: %v", data.PitchRange)
			log.Printf("  yaw_range:   %v", data.YawRange)

		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	}
}

func enableCORS(w http.ResponseWriter) {
	allowedOrigin := os.Getenv("ALLOWED_ORIGIN")
	if allowedOrigin == "" {
		allowedOrigin = "http://localhost:4040"
	}
	w.Header().Set("Access-Control-Allow-Origin", allowedOrigin)
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Accept, Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization")
	w.Header().Set("Access-Control-Expose-Headers", "Content-Length")
	w.Header().Set("Access-Control-Allow-Credentials", "true")
}

// CORS middleware wrapper
func corsMiddleware(handler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		enableCORS(w)

		// Handle preflight OPTIONS request
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		handler.ServeHTTP(w, r)
	})
}

func main() {
	// Load environment variables from .env file
	err := godotenv.Load()
	if err != nil {
		log.Printf("Error loading .env file: %v", err)
	}

	// Get MongoDB URI from environment variable or use default
	mongoURI := "mongodb://localhost:27017"
	if envURI := os.Getenv("MONGODB_URI"); envURI != "" {
		mongoURI = envURI
	}

	log.Println("Connecting to MongoDB")

	client, err := mongo.Connect(context.TODO(), options.Client().ApplyURI(mongoURI))
	if err != nil {
		log.Fatal("Failed to connect to MongoDB:", err)
	}

	// Test the connection
	err = client.Ping(context.TODO(), nil)
	if err != nil {
		log.Fatal("Failed to ping MongoDB:", err)
	}

	log.Println("Successfully connected to MongoDB")

	// Create a new mux to apply CORS middleware
	mux := http.NewServeMux()

	//Set up calibration
	calibrationsCollection := client.Database("testdb").Collection("calibrations")
	mux.HandleFunc("/api/v1/calibration", calibrationHandler(calibrationsCollection))
	mux.HandleFunc("/start-session", startSessionHandler)
	mux.HandleFunc("/start-calibration", startCalibrationHandler)
	mux.HandleFunc("/ai-agent/event", aiAgentHandler)

	// Serve static files (for our HTML page)
	mux.Handle("/", http.FileServer(http.Dir("./static/")))

	// Wrap everything with CORS middleware
	handler := corsMiddleware(mux)

	log.Println("Server starting on :4040")
	log.Println("API endpoints:")
	log.Println("  GET /api/v1/calibration?device_id=... (unprotected)")
	log.Println("  POST /api/v1/calibration (unprotected)")

	log.Println("  GET /api/v1/profiles/students (unprotected - for debugging)")
	log.Println("  POST /api/v1/profiles/students (JWT protected)")
	log.Println("  GET /api/v1/profiles/teachers (unprotected - for debugging)")
	log.Println("  POST /api/v1/profiles/teachers (JWT protected)")
	log.Println("  GET /api/v1/profiles/parents (unprotected - for debugging)")
	log.Println("  POST /api/v1/profiles/parents (JWT protected)")
	log.Println("CORS configured")
	log.Fatal(http.ListenAndServe(":4040", handler))
}
