import os, json, base64, subprocess

def handler(event):
    try:
        # Get input prompt from the RunPod payload
        input_data = event.get("input", {})
        prompt = input_data.get("prompt", "Default prompt")
        print(f"[RunPod] Received prompt: {prompt}")

        # Run your generate.py script
        result = subprocess.run(
            ["python3", "generate.py"],
            capture_output=True,
            text=True
        )

        # Check for saved MP4
        output_path = "/workspace/output/video.mp4"
        if os.path.exists(output_path):
            with open(output_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")


        return {
            "error": "⚠️ Video not found at /workspace/output/video.mp4",
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    test = {"input": {"prompt": "A cinematic test render"}}
    print(json.dumps(handler(test)))
