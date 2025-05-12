"""
LLM Server Configuration and Management

This module provides functionality to set up and manage different LLM servers
using vLLM, following SOLID principles and OOP best practices.

Dependencies:
- vllm
- pyyaml 
- pyngrok
- bitsandbytes>=0.45
- pydantic
- openai
"""

from abc import ABC, abstractmethod
import os
import json
import time
import subprocess
from typing import Dict, Optional, List
from dataclasses import dataclass
import yaml
import requests
import openai
from pyngrok import ngrok
from huggingface_hub import login

@dataclass
class ServerConfig:
    """Base configuration for LLM servers"""
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 8000
        self.uvicorn_log_level = "info"
        self.dtype = "float16"  # Changed from half to float16 for better compatibility
        self.gpu_memory_utilization = 0.85
        self.tensor_parallel_size = 1
        self.trust_remote_code = True
        self.max_model_len = 2048  # Reduced for faster startup
        self.quantization = None
        self.enforce_eager = True  # Added to prevent lazy loading
        self.block_size = 16  # Added for faster startup
        self.swap_space = 4  # Added to reduce memory pressure

class ModelConfig:
    """Model-specific configuration interface"""
    def __init__(self, base_config: ServerConfig):
        self.base_config = base_config
        
    @property
    @abstractmethod
    def config(self) -> Dict:
        """Return model-specific configuration"""
        pass

class MistralConfig(ModelConfig):
    def config(self) -> Dict:
        return {
            **self.base_config.__dict__,
            "model": "TheBloke/Mistral-7B-Instruct-v0.1-AWQ",
            "quantization": "awq",
            "max-model-len": 2048
        }

class QwenConfig(ModelConfig):
    def config(self) -> Dict:
        return {
            **self.base_config.__dict__,
            "model": "unsloth/Qwen3-8B-unsloth-bnb-4bit",
            "tokenizer": "unsloth/Qwen3-8B",
            "dtype": "float16",  # Changed from half to float16
            "max_model_len": 2048,  # Reduced from 3000 for faster startup
            "enforce_eager": True,
            "block_size": 16,
            "swap_space": 4
        }

class GemmaConfig(ModelConfig):
    def config(self) -> Dict:
        return {
            **self.base_config.__dict__,
            "model": "google/gemma-3-4b-it",
            "quantization": None,
            "max-model-len": 2048
        }

class AuthenticationService:
    """Handles authentication with external services"""
    def __init__(self, hf_token: str, ngrok_token: str):
        self.hf_token = hf_token
        self.ngrok_token = ngrok_token
        self.tunnel = None
        
    def setup_huggingface(self) -> None:
        """Set up Hugging Face authentication"""
        login(self.hf_token)
        print("✓ Hugging Face authentication successful")
        
    def setup_ngrok(self, port: int) -> str:
        """Set up ngrok tunnel and return public URL"""
        # Get all active tunnels
        tunnels = ngrok.get_tunnels()
        
        # Close all existing tunnels
        for tunnel in tunnels:
            print(f"Closing existing tunnel: {tunnel.public_url}")
            ngrok.disconnect(tunnel.public_url)
        
        # Set up new tunnel
        ngrok.set_auth_token(self.ngrok_token)
        self.tunnel = ngrok.connect(port, "http")
        print(f"🔗 New public URL → {self.tunnel.public_url}")
        return self.tunnel.public_url
        
    def cleanup(self):
        """Clean up ngrok tunnel"""
        if self.tunnel:
            try:
                ngrok.disconnect(self.tunnel.public_url)
                print("✓ Ngrok tunnel closed")
            except Exception as e:
                print(f"Error closing ngrok tunnel: {e}")

class ConfigurationManager:
    """Manages server configuration files"""
    @staticmethod
    def write_config(config: Dict, filename: str = "config.yaml") -> None:
        with open(filename, "w") as f:
            yaml.dump(config, f, sort_keys=False)
        print(f"✓ Configuration written to {filename}")
        
        with open(filename, "r") as f:
            print("\nConfiguration contents:")
            print(f.read())

class ServerManager:
    """Manages server lifecycle"""
    def __init__(self):
        self.server_process = None
        
    def launch(self, log_file: str) -> None:
        """Launch vLLM server in the background using direct shell commands"""
        # Kill any existing vLLM processes
        subprocess.run(['pkill', '-f', 'vllm serve'], stderr=subprocess.DEVNULL)
        
        # Start vLLM server in background using nohup
        cmd = f'nohup vllm serve --config config.yaml &> {log_file} &'
        subprocess.run(cmd, shell=True)
        
        print(f"✅ vLLM server starting... logs → {log_file}")
        
    def wait_for_server(self, port: int = 8000, max_retries: int = 60, retry_interval: int = 5) -> bool:
        """Wait for the vLLM server to be ready"""
        import time
        import socket
        
        for i in range(max_retries):
            try:
                # Try to connect to the server
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('127.0.0.1', port))
                sock.close()
                
                if result == 0:
                    print("✓ vLLM server is ready")
                    return True
                    
            except Exception:
                pass
                
            # Check if process is still running
            if i > 0 and i % 5 == 0:  # Check every 5 attempts
                try:
                    with open('qwen_server.log', 'r') as f:
                        log_content = f.read()
                        if 'error' in log_content.lower():
                            print("❌ Error detected in server logs:")
                            print(log_content[-500:])  # Print last 500 chars
                            return False
                        elif 'ready' in log_content.lower():
                            print("✓ Server reported ready in logs")
                            return True
                except Exception:
                    pass
                    
            print(f"Waiting for vLLM server to start... ({i+1}/{max_retries})")
            time.sleep(retry_interval)
            
        print("❌ vLLM server failed to start within the timeout period")
        return False
        
    def cleanup(self):
        """Clean up server resources"""
        try:
            # Kill vLLM process
            subprocess.run(['pkill', '-f', 'vllm serve'], stderr=subprocess.DEVNULL)
            print("✓ vLLM server stopped")
        except Exception as e:
            print(f"Error stopping vLLM server: {e}")

class LLMRequestHandler:
    """Handles LLM API requests"""
    @staticmethod
    def make_request(
        url: str,
        model: str,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False
    ) -> Dict:
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to make LLM request: {str(e)}")

class TextProcessor:
    """Processes text using LLM services"""
    def __init__(self, request_handler: LLMRequestHandler):
        self.request_handler = request_handler
        self.output_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "deutsch": {
                        "type": "string",
                        "description": "The original German full sentence."
                    },
                    "english": {
                        "type": "string",
                        "description": "The translated English full sentence."
                    }
                },
                "required": ["deutsch", "english"]
            },
            "description": "A list of translated full sentence pairs."
        }
    
    def process_german_text(self, text: str, api_base_url: str, model: str) -> str:
        
        if api_base_url.endswith("/completions"):
            api_base_url = api_base_url[:-len("/completions")]
        elif api_base_url.endswith("/chat/completions"):
            api_base_url = api_base_url[:-len("/chat/completions")]
        
        if not api_base_url.endswith("/v1"):
            if api_base_url.endswith("/"): 
                api_base_url += "v1"
            else: 
                api_base_url += "/v1"

        user_prompt = f"""Segment and translate this German transcript:

TRANSCRIPT:
{text}

RULES:
1. Find ALL LOGICAL BREAKS in the text (pauses, topic changes, sentence endings)
2. Create complete, meaningful sentences - as many as naturally exist
3. Each sentence should express one complete thought
4. Add capital letters and punctuation
5. Translate each sentence to English

EXAMPLE:
"das ist ein wichtiges Thema KI ist überall maschinelles Lernen verändert wie wir arbeiten neue Tools müssen verstanden werden"

GOOD OUTPUT:
[
  {{
    "deutsch": "Das ist ein wichtiges Thema.",
    "english": "This is an important topic."
  }},
  {{
    "deutsch": "KI ist überall.",
    "english": "AI is everywhere."
  }},
  {{
    "deutsch": "Maschinelles Lernen verändert, wie wir arbeiten.",
    "english": "Machine learning changes how we work."
  }},
  {{
    "deutsch": "Neue Tools müssen verstanden werden.",
    "english": "New tools need to be understood."
  }}
]

BAD OUTPUT (don't do this - fragments, not complete thoughts):
[
  {{
    "deutsch": "Das ist ein wichtiges.",
    "english": "This is an important."
  }},
  {{
    "deutsch": "KI ist.",
    "english": "AI is."
  }}
]

BAD OUTPUT (don't do this - everything in one sentence):
[
  {{
    "deutsch": "Das ist ein wichtiges Thema KI ist überall maschinelles Lernen verändert wie wir arbeiten neue Tools müssen verstanden werden.",
    "english": "This is an important topic AI is everywhere machine learning changes how we work new tools need to be understood."
  }}
]

Extract ALL natural sentences, each expressing ONE complete thought.
"""
        client = openai.OpenAI(base_url=api_base_url, api_key="dummy_key")

        try:
            # Add retry logic for the API call
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You reconstruct complete German sentences from transcript lines, then translate them to English. Respond in JSON format."},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.1,
                        max_tokens=1000,
                        extra_body={
                            "guided_json": self.output_schema,
                            "enable_thinking": False
                        }
                    )
                    response_content = completion.choices[0].message.content
                    return json.dumps(json.loads(response_content), indent=2)
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise

        except openai.APIConnectionError as e:
            raise Exception(f"CONNECTION ERROR: {e}. Check URL and if vLLM server is running.")
        except json.JSONDecodeError as e:
            raise Exception(f"JSON PARSING ERROR: {e}. Model output might not be valid JSON.")
        except Exception as e:
            raise Exception(f"UNEXPECTED ERROR: {e}")

class LLMServerFactory:
    """Creates and configures different types of LLM servers"""
    def __init__(
        self,
        auth_service: AuthenticationService,
        config_manager: ConfigurationManager,
        server_manager: ServerManager
    ):
        self.auth_service = auth_service
        self.config_manager = config_manager
        self.server_manager = server_manager
        
    def create_server(self, model_config: ModelConfig, log_file: str) -> str:
        """Create and launch a server with the given configuration"""
        try:
            # Write config and start server
            self.config_manager.write_config(model_config.config())
            self.server_manager.launch(log_file)
            
            # Wait for server to be ready
            if not self.server_manager.wait_for_server(model_config.base_config.port):
                raise Exception("Failed to start vLLM server")
            
            # Set up ngrok tunnel
            return self.auth_service.setup_ngrok(model_config.base_config.port)
            
        except Exception as e:
            # Ensure cleanup on error
            self.server_manager.cleanup()
            self.auth_service.cleanup()
            raise e

def cleanup_ngrok():
    """Clean up all ngrok connections."""
    try:
        # Get and close all tunnels
        tunnels = ngrok.get_tunnels()
        for tunnel in tunnels:
            print(f"Closing tunnel: {tunnel.public_url}")
            ngrok.disconnect(tunnel.public_url)
            
        print("✅ All ngrok connections cleaned up")
    except Exception as e:
        print(f"Error during cleanup: {e}")

def start_server(log_file: str = "vllm_server.log", model_name: str = None) -> None:
    """
    Start a vLLM server properly in the background, even in Jupyter
    """
    import os
    import signal
    import platform
    
    # Kill any existing vLLM processes first
    print("Stopping any existing vLLM processes...")
    kill_cmd = 'pkill -f "vllm serve" 2>/dev/null || true'
    subprocess.run(kill_cmd, shell=True, capture_output=True)
    
    # Create log file directory if needed
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Touch the log file to ensure it exists
    with open(log_file, 'w') as f:
        f.write(f"Starting vLLM server at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # In Jupyter, we need special handling to truly run in background
    # This creates a new session and ensures the process is fully detached
    print(f"Starting vLLM server in background, output will be in {log_file}...")
    
    # Start the server using a different approach that guarantees background operation
    if platform.system() == "Windows":
        # Windows version
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        DETACHED_PROCESS = 0x00000008
        process = subprocess.Popen(
            ['vllm', 'serve', '--config', 'config.yaml'],
            stdout=open(log_file, 'a'),
            stderr=subprocess.STDOUT,
            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        )
    else:
        # Unix version - use double fork method to fully detach
        try:
            # Write a launcher script that will run the actual command
            launcher_script = "vllm_launcher.sh"
            with open(launcher_script, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write(f"vllm serve --config config.yaml > {log_file} 2>&1 &\n")
            
            # Make it executable
            os.chmod(launcher_script, 0o755)
            
            # Execute the launcher script
            subprocess.run(f"bash ./{launcher_script}", shell=True)
            
            # Remove the launcher script
            os.remove(launcher_script)
        except Exception as e:
            print(f"Error launching script: {str(e)}")
    
    # Verify the process started
    time.sleep(2)  # Give it a moment to start
    check_cmd = 'ps aux | grep "vllm serve" | grep -v grep'
    check_process = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
    
    if check_process.stdout:
        print("✅ vLLM server process found and running in background")
    else:
        print("⚠️ Warning: vLLM server process not found after starting")

    if model_name:
        print(f"✅ vLLM ({model_name}) starting in background... logs → {log_file}")
    else:
        print(f"✅ vLLM server starting in background... logs → {log_file}")
        
    # Display the first few lines of the log
    print("\nChecking log file...")
    subprocess.run(f"tail -n 10 {log_file}", shell=True)

def setup_ngrok(port: int = 8000) -> str:
    """Set up ngrok tunnel and return public URL"""
    # Get all active tunnels
    tunnels = ngrok.get_tunnels()
    
    # Close all existing tunnels
    for tunnel in tunnels:
        print(f"Closing existing tunnel: {tunnel.public_url}")
        ngrok.disconnect(tunnel.public_url)
    
    # Set up new tunnel
    auth_token = os.getenv("NGROK_AUTH_TOKEN")
    if not auth_token:
        raise ValueError("NGROK_AUTH_TOKEN environment variable is required")
    ngrok.set_auth_token(auth_token)
    public_url = ngrok.connect(port, "http")
    print(f"🔗 New public URL → {public_url.public_url}")
    return public_url.public_url

def cleanup():
    """Clean up resources"""
    # Kill any vLLM processes
    subprocess.run('pkill -f "vllm serve" 2>/dev/null || true', shell=True)
    
    # Close all tunnels
    tunnels = ngrok.get_tunnels()
    for tunnel in tunnels:
        ngrok.disconnect(tunnel.public_url)
    
    print("✓ Cleanup complete")

def main():
    """Main execution function - this can be run both in script and notebook"""
    try:
        # Set up Hugging Face authentication
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable is required")
        login(hf_token)
        print("✓ Hugging Face authentication successful")
        
        # Create server config
        base_config = ServerConfig()
        qwen_config = QwenConfig(base_config)
        
        # Write config
        with open("config.yaml", "w") as f:
            yaml.dump(qwen_config.config(), f, sort_keys=False)
        print("✓ Configuration written to config.yaml")
        
        # Print config contents
        with open("config.yaml", "r") as f:
            print("\nConfiguration contents:")
            print(f.read())
        
        # Start server
        start_server("qwen_server.log", "Qwen3-8B")
        
        # Allow time for server to start
        print("Waiting 10 seconds for server to start...")
        time.sleep(10)
        
        # Check log file again
        print("\nChecking log file after waiting:")
        subprocess.run("tail -n 20 qwen_server.log", shell=True)
        
        # Set up ngrok tunnel with authentication
        ngrok_auth_token = os.getenv("NGROK_AUTH_TOKEN")
        if not ngrok_auth_token:
            raise ValueError("NGROK_AUTH_TOKEN environment variable is required")
        ngrok.set_auth_token(ngrok_auth_token)
        server_url = setup_ngrok(8000)
        
        print("\nServer is running in background!")
        print(f"Public URL: {server_url}")
        print(f"API Endpoint: {server_url}/v1/completions")
        print("Server will continue running in background even after this cell completes")
        
        # Process example text (keep the comment for reference)
        """
        german_text = '''
        das ist ein Thema das an keinem von uns
        Vorbeigehen sollte ai agents warum sind
        die relevant wenn wir an die
        ki-anwendung denken mit dem wir heute
        arbeiten wie unterscheidet sich ein ai
        Agent von dem was ich in jgpt mache oder
        Copilot und was hat das mit meiner
        Arbeit zu
        [Musik]
        '''

        vllm_url = f"{server_url}/v1/completions"
        result = processor.process_german_text(
            german_text,
            vllm_url,
            "unsloth/Qwen3-8B-unsloth-bnb-4bit"
        )
        print(result)
        """
        
        # Don't keep the main script running - this allows the cell to complete
        # and return control to the user while the server runs in background
        return server_url
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"\nError: {e}")
        raise
    finally:
        # We don't want to call cleanup here - it would kill the server!
        # Only uncomment this if you want to stop the server when the cell completes
        # cleanup()
        pass

if __name__ == "__main__":
    main()






# Keep the above i need it for later 

# import importlib
# import lm_mistral_server
# importlib.reload(lm_mistral_server)
# from lm_mistral_server import *


# Your German text
# german_text = """
# das ist ein Thema das an keinem von uns
# Vorbeigehen sollte ai agents warum sind
# die relevant wenn wir an die
# ki-anwendung denken mit dem wir heute
# arbeiten wie unterscheidet sich ein ai
# Agent von dem was ich in jgpt mache oder
# Copilot und was hat das mit meiner
# Arbeit zu
# [Musik]
# """


# server_url = setup_ngrok(8000)

# # Create the processor
# request_handler = LLMRequestHandler()
# processor = TextProcessor(request_handler)

# # Process the text
# vllm_url = f"{server_url}/v1/completions"
# result = processor.process_german_text(
#     german_text,
#     vllm_url,
#     "unsloth/Qwen3-8B-unsloth-bnb-4bit"
# )
# print(result)