import os
import sys
import re
import time
import subprocess

# Log file targets configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
LOG_CONFIGS = [
    {
        "name": "ResNet50 + CE",
        "path": os.path.join(script_dir, "../logs/resnet50_ce.log")
    },
    {
        "name": "ConvNeXt-Small + CORAL",
        "path": os.path.join(script_dir, "../logs/convnext_small_coral.log")
    },
    {
        "name": "ConvNeXt-Small + CORN",
        "path": os.path.join(script_dir, "../logs/convnext_small_corn.log")
    }
]

# ANSI colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"

def get_gpu_info():
    """Queries nvidia-smi for VRAM utilization across all devices."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,utilization.gpu,name", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2
        )
        if res.returncode == 0:
            gpu_lines = res.stdout.strip().split("\n")
            outputs = []
            for idx, line in enumerate(gpu_lines):
                parts = line.split(",")
                if len(parts) >= 4:
                    total_mem = float(parts[0].strip())
                    used_mem = float(parts[1].strip())
                    util = parts[2].strip()
                    name = parts[3].strip()
                    pct_mem = (used_mem / total_mem) * 100
                    outputs.append(f"GPU {idx} ({name}): Util {util}% | VRAM {used_mem/1024:.1f}GB/{total_mem/1024:.1f}GB ({pct_mem:.1f}%)")
            return "\n ".join(outputs)
    except Exception:
        pass
    return "GPU Info: nvidia-smi query unavailable"

def parse_log(log_path):
    """Parses a log file and returns details like status, epoch, loss, val_kappa, speed, and ETC."""
    if not os.path.exists(log_path):
        return {
            "status": "WAITING",
            "color": YELLOW,
            "epoch": "N/A",
            "progress": "N/A",
            "loss": "N/A",
            "val_kappa": "N/A",
            "speed": "N/A",
            "etc": "N/A",
            "raw_speed": 0.0,
            "error_msg": ""
        }

    with open(log_path, 'r', errors='ignore') as f:
        lines = f.readlines()

    if not lines:
        return {
            "status": "STARTING",
            "color": CYAN,
            "epoch": "N/A",
            "progress": "N/A",
            "loss": "N/A",
            "val_kappa": "N/A",
            "speed": "N/A",
            "etc": "N/A",
            "raw_speed": 0.0,
            "error_msg": ""
        }

    # Inspect the end of log file for crashes
    last_content = "".join(lines[-40:])
    
    is_oom = any(x in last_content for x in ["Out of memory", "CUDA out of memory", "OutOfMemoryError"])
    is_crashed = any(x in last_content for x in ["Traceback", "RuntimeError:", "Exception:", "AttributeError:"])
    
    # Check for final completion signals
    is_finished = any(x in last_content for x in [
        "completely finalized",
        "ALL 4 ARCHITECTURES SUCCESSFULLY TRAINED",
        "Early stopping hit"
    ])

    epoch = "N/A"
    loss = "N/A"
    val_kappa = "N/A"
    progress = "N/A"
    speed = "N/A"
    etc = "N/A"
    raw_speed = 0.0
    error_msg = ""

    # Regular expressions to parse training logging summary and tqdm outputs
    # Summary log: Epoch 12/30 | Loss: 0.4321 | Val Kappa: 0.5891
    log_pat = re.compile(r"Epoch (\d+)/(\d+) \| Loss: ([\d\.]+) \| Val Kappa: ([\-\d\.]+)")
    # tqdm progress pattern: 45%|███       | 740/1642 [00:45<05:32, 4.20it/s, loss=0.4321]
    tqdm_pat = re.compile(r"(\d+)%\|.*\| (\d+/\d+) \[(.*?)<(.*?), (.*?)\]")
    
    last_log_epoch = None
    last_log_loss = None
    last_log_kappa = None

    for line in reversed(lines):
        # 1. Parse Completed Epoch Logging
        log_match = log_pat.search(line)
        if log_match:
            if last_log_epoch is None:
                last_log_epoch = int(log_match.group(1))
                last_log_loss = log_match.group(3)
                last_log_kappa = log_match.group(4)
        
        # 2. Parse Batch Progress
        tqdm_match = tqdm_pat.search(line)
        if tqdm_match and progress == "N/A":
            pct = tqdm_match.group(1)
            batches = tqdm_match.group(2)
            elapsed = tqdm_match.group(3)
            rem = tqdm_match.group(4)
            spd_info = tqdm_match.group(5)
            
            # Parse speed and loss from the trailing info
            spd_parts = spd_info.split(",")
            spd = spd_parts[0].strip()
            
            progress = f"{pct}% (Batch {batches})"
            speed = spd
            
            # Use batch-level loss for the Train Loss field
            if len(spd_parts) > 1 and "loss=" in spd_parts[1]:
                loss = spd_parts[1].replace("loss=", "").strip()
            
            # Parse raw speed value
            try:
                if "it/s" in spd:
                    raw_speed = float(re.findall(r"[\d\.]+", spd)[0])
                elif "s/it" in spd:
                    s_per_it = float(re.findall(r"[\d\.]+", spd)[0])
                    if s_per_it > 0:
                        raw_speed = 1.0 / s_per_it
            except Exception:
                pass

    # Extract current epoch and total epochs
    # Look backward for current active epoch index
    epoch_pat = re.compile(r"Epoch (\d+)/(\d+)")
    epoch_num = None
    total_epochs = "30"
    for line in reversed(lines):
        m = epoch_pat.search(line)
        if m:
            epoch_num = int(m.group(1))
            total_epochs = m.group(2)
            break

    if epoch_num:
        epoch = f"{epoch_num}/{total_epochs}"
    elif last_log_epoch:
        # If we have finished an epoch but a new one hasn't registered in text yet
        epoch = f"{last_log_epoch}/{total_epochs}"
    else:
        epoch = f"1/{total_epochs}"

    # Set metrics
    if last_log_loss:
        loss = last_log_loss
    if last_log_kappa:
        val_kappa = last_log_kappa

    # Estimate Total ETC
    if raw_speed > 0.0 and epoch_num and progress != "N/A":
        try:
            total_epochs_val = int(total_epochs)
            batch_parts = re.findall(r"\d+", progress)
            if len(batch_parts) >= 2:
                curr_batch = int(batch_parts[0])
                total_batches = int(batch_parts[1])
            else:
                curr_batch = 0
                total_batches = 1642  # default fallback
                
            rem_batches_epoch = total_batches - curr_batch
            sec_rem_epoch = rem_batches_epoch / raw_speed
            
            rem_epochs = total_epochs_val - epoch_num
            sec_per_epoch = total_batches / raw_speed
            total_rem_sec = sec_rem_epoch + (rem_epochs * sec_per_epoch)
            
            h = int(total_rem_sec // 3600)
            m = int((total_rem_sec % 3600) // 60)
            s = int(total_rem_sec % 60)
            if h > 0:
                etc = f"{h}h {m}m"
            else:
                etc = f"{m}m {s}s"
        except Exception:
            etc = "Calculating..."
    else:
        etc = "--"

    # Resolve Status & Colors
    if is_oom:
        status = "OOM CRASH"
        color = RED
        error_msg = "CUDA Out Of Memory"
    elif is_crashed:
        status = "CRASHED"
        color = RED
        # Grab last lines for error detail
        for l in reversed(lines):
            l_strip = l.strip()
            if l_strip and (":" in l_strip or "Error" in l_strip or "Exception" in l_strip):
                error_msg = l_strip
                break
        if not error_msg:
            error_msg = "Unknown execution error"
    elif is_finished:
        status = "FINISHED"
        color = GREEN
        epoch = f"{total_epochs}/{total_epochs}"
        progress = "100%"
        etc = "Done"
    else:
        status = "TRAINING"
        color = CYAN

    return {
        "status": status,
        "color": color,
        "epoch": epoch,
        "progress": progress,
        "loss": loss,
        "val_kappa": val_kappa,
        "speed": speed,
        "etc": etc,
        "raw_speed": raw_speed,
        "error_msg": error_msg
    }

def print_dashboard():
    # Clear screen
    sys.stdout.write("\033[H\033[2J")
    sys.stdout.flush()

    gpu_status = get_gpu_info()
    results = []
    
    for cfg in LOG_CONFIGS:
        res = parse_log(cfg["path"])
        res["name"] = cfg["name"]
        res["log"] = cfg["path"]
        results.append(res)

    print("=" * 80)
    print(f" {BOLD}{MAGENTA}🔥 PARALLEL MAMMOGRAPHY TRAINING DASHBOARD{RESET}")
    print(f" System Time: {time.strftime('%Y-%m-%d %H:%M:%S')} | Refresh Interval: 5s")
    print("=" * 80)
    print(f" {BOLD}GPU Status:{RESET}")
    print(f" {gpu_status}")
    print("-" * 80)

    active_speeds = []
    for res in results:
        # Print info card for each model
        status_line = f" {BOLD}[{res['name']}]{RESET} - Log: {res['log']}"
        print(status_line)
        print(f"   Status     : {res['color']}{BOLD}{res['status']}{RESET}")
        
        if res['status'] in ["OOM CRASH", "CRASHED"]:
            print(f"   Error      : {RED}{res['error_msg']}{RESET}")
            
        print(f"   Epoch      : {res['epoch']}")
        print(f"   Progress   : {res['progress']}")
        print(f"   Train Loss : {res['loss']}")
        print(f"   Val Kappa  : {res['val_kappa']}")
        print(f"   Speed      : {res['speed']}")
        print(f"   Est. ETC   : {res['etc']}")
        print()

        if res['status'] == "TRAINING" and res['raw_speed'] > 0:
            active_speeds.append((res['name'], res['raw_speed']))

    print("-" * 80)
    print(f" {BOLD}🏆 PERFORMANCE COMPARISON:{RESET}")
    if len(active_speeds) >= 1:
        sorted_speeds = sorted(active_speeds, key=lambda x: x[1], reverse=True)
        print(f"   ⚡ Fastest: {GREEN}{sorted_speeds[0][0]}{RESET} ({sorted_speeds[0][1]:.2f} it/s)")
        if len(sorted_speeds) > 1:
            print(f"   🐢 Slowest: {YELLOW}{sorted_speeds[-1][0]}{RESET} ({sorted_speeds[-1][1]:.2f} it/s)")
    else:
        print("   No actively running training sessions showing benchmark speeds.")

    print("\n" + "-" * 80)
    print(f" {BOLD}🛑 EMERGENCY ACTIONS:{RESET}")
    print(f"   Kill all training processes one-liner:")
    print(f"     {BOLD}{RED}pkill -9 -f \"training_script.py|resnet50_ce|convnext_small\"{RESET}")
    print(f"   Or run this script with --kill flag:")
    print(f"     {BOLD}python3 monitor_training.py --kill{RESET}")
    print("=" * 80)

if __name__ == '__main__':
    # Handle the --kill flag
    if len(sys.argv) > 1 and sys.argv[1] == "--kill":
        print(f"{BOLD}{RED}Terminating all training processes...{RESET}")
        cmd = "pkill -9 -f 'training_script.py|resnet50_ce|convnext_small'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print("Kill signal sent via pkill.")
        sys.exit(0)

    try:
        while True:
            print_dashboard()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nExiting dashboard monitor. Goodbye!")
