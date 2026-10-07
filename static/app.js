// DOM Elements
const connectionStatusEl = document.getElementById("connection-status");
const mappingPctEl = document.getElementById("mapping-pct");
const poseErrorEl = document.getElementById("pose-error");

const btnManual = document.getElementById("btn-manual");
const btnAuto = document.getElementById("btn-auto");
const mapPresetSel = document.getElementById("map-preset");
const btnResetOdom = document.getElementById("btn-reset-odom");
const btnClearWalls = document.getElementById("btn-clear-walls");

const lidarBeamsSlider = document.getElementById("lidar-beams");
const lidarRangeSlider = document.getElementById("lidar-range");
const sensorNoiseSwitch = document.getElementById("sensor-noise");

const beamsValEl = document.getElementById("beams-val");
const rangeValEl = document.getElementById("range-val");

const telXEl = document.getElementById("tel-x");
const telYEl = document.getElementById("tel-y");
const telThetaEl = document.getElementById("tel-theta");
const telVEl = document.getElementById("tel-v");
const telWEl = document.getElementById("tel-w");

// Canvases
const canvasTrue = document.getElementById("canvas-true");
const canvasSLAM = document.getElementById("canvas-slam");
const canvasRadar = document.getElementById("canvas-radar");

const ctxTrue = canvasTrue.getContext("2d");
const ctxSLAM = canvasSLAM.getContext("2d");
const ctxRadar = canvasRadar.getContext("2d");

// App State
let socket = null;
let currentPreset = "office";
let isAutoExploration = false;
let width = 800;
let height = 600;

// Mouse drawing state for Canvas True (draw walls)
let isDrawing = false;
let drawStartX = 0;
let drawStartY = 0;
let drawEndX = 0;
let drawEndY = 0;

// Key pressed state for WASD robot control
const keys = { w: false, a: false, s: false, d: false };

// Connect WebSockets
function connect() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host || "localhost:8000";
    const wsUrl = `${protocol}//${host}/ws`;

    console.log(`Connecting to WebSocket: ${wsUrl}`);
    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
        console.log("WebSocket connection established.");
        connectionStatusEl.innerText = "CONNECTED";
        connectionStatusEl.className = "stat-value text-green";
    };

    socket.onclose = () => {
        console.warn("WebSocket connection lost. Retrying in 2 seconds...");
        connectionStatusEl.innerText = "OFFLINE";
        connectionStatusEl.className = "stat-value text-red";
        setTimeout(connect, 2000);
    };

    socket.onerror = (err) => {
        console.error("WebSocket encountered an error:", err);
    };

    socket.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === "init") {
            handleInit(msg);
        } else {
            handleUpdate(msg);
        }
    };
}

// Handle Init Message
function handleInit(msg) {
    width = msg.width;
    height = msg.height;
    canvasTrue.width = width;
    canvasTrue.height = height;
    canvasSLAM.width = width;
    canvasSLAM.height = height;
}

// Handle State Update from Python
function handleUpdate(state) {
    // 1. Update Telemetry text values
    const [tx, ty, ttheta] = state.true_pose;
    const [ox, oy, otheta] = state.odo_pose;
    
    telXEl.innerText = `${ox.toFixed(1)} px`;
    telYEl.innerText = `${oy.toFixed(1)} px`;
    telThetaEl.innerText = `${otheta.toFixed(2)} rad`;
    telVEl.innerText = `${state.v.toFixed(1)} px/s`;
    telWEl.innerText = `${state.w.toFixed(2)} rad/s`;

    // Mapping Progress
    mappingPctEl.innerText = `${(state.explored_ratio * 100).toFixed(1)}%`;
    
    // Pose drift error
    const errX = tx - ox;
    const errY = ty - oy;
    const errorDist = Math.hypot(errX, errY);
    poseErrorEl.innerText = `${errorDist.toFixed(1)} px`;
    if (errorDist > 30.0) {
        poseErrorEl.className = "stat-value text-orange";
    } else {
        poseErrorEl.className = "stat-value text-cyan";
    }

    // Toggle navigation mode button UI if state changes internally
    isAutoExploration = state.auto_explore;
    if (isAutoExploration) {
        btnAuto.classList.add("active");
        btnManual.classList.remove("active");
    } else {
        btnManual.classList.add("active");
        btnAuto.classList.remove("active");
    }

    // 2. Render viewports
    renderTrueSim(state);
    renderSLAMMap(state);
    renderRadarPlot(state);
}

// Render TRUE SIMULATION Viewport
function renderTrueSim(state) {
    const ctx = ctxTrue;
    ctx.clearRect(0, 0, width, height);

    // Grid pattern background
    drawGrid(ctx);

    // True Walls
    ctx.strokeStyle = "rgba(74, 85, 104, 0.6)";
    ctx.lineWidth = 3;
    state.true_walls.forEach(([x1, y1, x2, y2]) => {
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
    });

    // Draw custom user wall being dragged
    if (isDrawing) {
        ctx.strokeStyle = "rgba(255, 159, 28, 0.8)";
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(drawStartX, drawStartY);
        ctx.lineTo(drawEndX, drawEndY);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Draw LiDAR Ray sweeps
    const [rx, ry, rtheta] = state.true_pose;
    const ranges = state.lidar_ranges;
    const angles = state.lidar_angles;

    ctx.lineWidth = 1;
    for (let i = 0; i < ranges.length; i++) {
        const angle = angles[i] + rtheta;
        const dist = ranges[i];
        
        // Ray line
        ctx.strokeStyle = "rgba(255, 51, 102, 0.12)";
        ctx.beginPath();
        ctx.moveTo(rx, ry);
        const hx = rx + dist * Math.cos(angle);
        const hy = ry + dist * Math.sin(angle);
        ctx.lineTo(hx, hy);
        ctx.stroke();

        // Hit point
        if (dist < lidarRangeSlider.value) {
            ctx.fillStyle = "rgba(255, 51, 102, 0.8)";
            ctx.beginPath();
            ctx.arc(hx, hy, 2, 0, 2 * Math.PI);
            ctx.fill();
        }
    }

    // Draw Target crosshair if present
    if (state.target) {
        drawTarget(ctx, state.target[0], state.target[1]);
    }

    // Draw True Robot (red/cyan glowing circle)
    drawRobot(ctx, rx, ry, rtheta, "#ff3366", "rgba(255, 51, 102, 0.3)");
}

// Render SLAM Occupancy Grid Map
function renderSLAMMap(state) {
    const ctx = ctxSLAM;
    ctx.clearRect(0, 0, width, height);

    const cols = state.grid_cols;
    const rows = state.grid_rows;
    const cell_size = state.grid_cell_size;
    const slam_grid = state.slam_grid;

    // Direct pixel rendering with ImageData for high 60fps performance
    // We render at the native grid size, then draw onto the full canvas.
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = cols;
    tempCanvas.height = rows;
    const tempCtx = tempCanvas.getContext("2d");
    const imgData = tempCtx.createImageData(cols, rows);

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const idx = r * cols + c;
            const prob = slam_grid[idx];
            let r_col, g_col, b_col, alpha;

            if (prob > 65) {
                // Occupied obstacle cell (Glowing Cyan)
                r_col = 0; g_col = 242; b_col = 254; alpha = 255;
            } else if (prob < 40) {
                // Explored Free cell (Clear White/Light blue)
                r_col = 240; g_col = 245; b_col = 255; alpha = 230;
            } else {
                // Unknown space cell (Dark background)
                r_col = 15; g_col = 22; b_col = 38; alpha = 255;
            }

            const pixelIdx = (r * cols + c) * 4;
            imgData.data[pixelIdx] = r_col;
            imgData.data[pixelIdx + 1] = g_col;
            imgData.data[pixelIdx + 2] = b_col;
            imgData.data[pixelIdx + 3] = alpha;
        }
    }
    
    tempCtx.putImageData(imgData, 0, 0);
    
    // Scale and draw without anti-aliasing for a clean matrix pixel-art look
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tempCanvas, 0, 0, width, height);
    ctx.imageSmoothingEnabled = true;

    // Draw A* Planned Path
    if (state.path && state.path.length > 0) {
        ctx.strokeStyle = "#05ffc5";
        ctx.lineWidth = 2.5;
        ctx.shadowColor = "#05ffc5";
        ctx.shadowBlur = 6;
        ctx.beginPath();
        ctx.moveTo(state.path[0][0], state.path[0][1]);
        for (let i = 1; i < state.path.length; i++) {
            ctx.lineTo(state.path[i][0], state.path[i][1]);
        }
        ctx.stroke();
        ctx.shadowBlur = 0; // Reset shadow
    }

    // Draw Target crosshair
    if (state.target) {
        drawTarget(ctx, state.target[0], state.target[1]);
    }

    // Draw Estimated Robot (yellow glowing circle based on Odometry pose)
    const [ox, oy, otheta] = state.odo_pose;
    drawRobot(ctx, ox, oy, otheta, "#00f2fe", "rgba(0, 242, 254, 0.3)");
}

// Render RADAR Polar Scan Sweep
function renderRadarPlot(state) {
    const ctx = ctxRadar;
    const w = canvasRadar.width;
    const h = canvasRadar.height;
    const cx = w / 2;
    const cy = h / 2;
    const r_max = cx - 10;

    ctx.clearRect(0, 0, w, h);

    // Draw concentric radar lines
    ctx.strokeStyle = "rgba(0, 242, 254, 0.15)";
    ctx.lineWidth = 1;
    for (let radius = 25; radius <= r_max; radius += 30) {
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
        ctx.stroke();
    }

    // Diagonal sweep coordinates axes
    ctx.beginPath();
    ctx.moveTo(cx - r_max, cy); ctx.lineTo(cx + r_max, cy);
    ctx.moveTo(cx, cy - r_max); ctx.lineTo(cx, cy + r_max);
    ctx.stroke();

    // Plot scan readings
    const ranges = state.lidar_ranges;
    const angles = state.lidar_angles;
    const max_range = parseFloat(lidarRangeSlider.value);

    ctx.fillStyle = "rgba(5, 255, 197, 0.75)";
    ctx.strokeStyle = "rgba(5, 255, 197, 0.3)";
    ctx.beginPath();

    for (let i = 0; i < ranges.length; i++) {
        const angle = angles[i] - Math.PI / 2; // Orient radar upwards
        const fraction = ranges[i] / max_range;
        const r_val = fraction * r_max;

        const px = cx + r_val * Math.cos(angle);
        const py = cy + r_val * Math.sin(angle);

        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);

        // Individual dots
        ctx.fillRect(px - 1.5, py - 1.5, 3, 3);
    }
    ctx.stroke();
}

// Helper Canvas Draw Utilities
function drawGrid(ctx) {
    ctx.strokeStyle = "rgba(255, 255, 255, 0.02)";
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 40) {
        ctx.beginPath();
        ctx.moveTo(x, 0); ctx.lineTo(x, height);
        ctx.stroke();
    }
    for (let y = 0; y < height; y += 40) {
        ctx.beginPath();
        ctx.moveTo(0, y); ctx.lineTo(width, y);
        ctx.stroke();
    }
}

function drawRobot(ctx, rx, ry, rtheta, primaryColor, glowColor) {
    // Glow ring
    ctx.fillStyle = glowColor;
    ctx.beginPath();
    ctx.arc(rx, ry, 15, 0, 2 * Math.PI);
    ctx.fill();

    // Robot body outline
    ctx.strokeStyle = primaryColor;
    ctx.lineWidth = 2.5;
    ctx.fillStyle = "#0c101d";
    ctx.beginPath();
    ctx.arc(rx, ry, 12, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();

    // Heading line pointer
    ctx.strokeStyle = primaryColor;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(rx, ry);
    ctx.lineTo(rx + 15 * Math.cos(rtheta), ry + 15 * Math.sin(rtheta));
    ctx.stroke();
}

function drawTarget(ctx, tx, ty) {
    ctx.strokeStyle = "#ff9f1c";
    ctx.lineWidth = 2;
    ctx.shadowColor = "#ff9f1c";
    ctx.shadowBlur = 6;
    
    // Crosshair target
    ctx.beginPath();
    ctx.arc(tx, ty, 8, 0, 2 * Math.PI);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.moveTo(tx - 12, ty); ctx.lineTo(tx + 12, ty);
    ctx.moveTo(tx, ty - 12); ctx.lineTo(tx, ty + 12);
    ctx.stroke();
    
    ctx.shadowBlur = 0;
}

// Input Event Listeners
function setupInputListeners() {
    // 1. WASD Manual robot driving
    window.addEventListener("keydown", (e) => {
        const key = e.key.toLowerCase();
        if (["w", "a", "s", "d"].includes(key)) {
            keys[key] = true;
            sendManualControl();
        }
    });

    window.addEventListener("keyup", (e) => {
        const key = e.key.toLowerCase();
        if (["w", "a", "s", "d"].includes(key)) {
            keys[key] = false;
            sendManualControl();
        }
    });

    function sendManualControl() {
        if (isAutoExploration) return;
        
        let linear = 0.0;
        let angular = 0.0;
        
        if (keys.w) linear += 100.0;
        if (keys.s) linear -= 60.0;
        if (keys.a) angular -= 1.8;
        if (keys.d) angular += 1.8;

        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(jsonMsg("manual_control", { v: linear, w: angular }));
        }
    }

    // 2. Click to set target on SLAM canvas
    canvasSLAM.addEventListener("click", (e) => {
        const rect = canvasSLAM.getBoundingClientRect();
        const clickX = ((e.clientX - rect.left) / rect.width) * width;
        const clickY = ((e.clientY - rect.top) / rect.height) * height;

        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(jsonMsg("set_target", { x: clickX, y: clickY }));
        }
    });

    // 3. True Canvas draw walls (MouseDown, MouseMove, MouseUp)
    canvasTrue.addEventListener("mousedown", (e) => {
        const rect = canvasTrue.getBoundingClientRect();
        drawStartX = ((e.clientX - rect.left) / rect.width) * width;
        drawStartY = ((e.clientY - rect.top) / rect.height) * height;
        isDrawing = true;
        drawEndX = drawStartX;
        drawEndY = drawStartY;
    });

    canvasTrue.addEventListener("mousemove", (e) => {
        if (!isDrawing) return;
        const rect = canvasTrue.getBoundingClientRect();
        drawEndX = ((e.clientX - rect.left) / rect.width) * width;
        drawEndY = ((e.clientY - rect.top) / rect.height) * height;
    });

    canvasTrue.addEventListener("mouseup", () => {
        if (!isDrawing) return;
        isDrawing = false;
        // If segment has length, send add_wall command
        const len = Math.hypot(drawEndX - drawStartX, drawEndY - drawStartY);
        if (len > 10.0 && socket && socket.readyState === WebSocket.OPEN) {
            socket.send(jsonMsg("add_wall", {
                x1: drawStartX, y1: drawStartY,
                x2: drawEndX, y2: drawEndY
            }));
        }
    });

    // 4. UI Side Panel Listeners
    btnManual.addEventListener("click", () => {
        isAutoExploration = false;
        btnManual.classList.add("active");
        btnAuto.classList.remove("active");
        sendConfig({ type: "toggle_auto_exploration", value: false });
    });

    btnAuto.addEventListener("click", () => {
        isAutoExploration = true;
        btnAuto.classList.add("active");
        btnManual.classList.remove("active");
        sendConfig({ type: "toggle_auto_exploration", value: true });
    });

    mapPresetSel.addEventListener("change", (e) => {
        currentPreset = e.target.value;
        sendConfig({ type: "load_preset", name: currentPreset });
    });

    btnResetOdom.addEventListener("click", () => {
        sendConfig({ type: "reset_odom" });
    });

    btnClearWalls.addEventListener("click", () => {
        sendConfig({ type: "clear_walls" });
    });

    // Live slider updating
    lidarBeamsSlider.addEventListener("input", (e) => {
        beamsValEl.innerText = e.target.value;
        syncSensorConfig();
    });

    lidarRangeSlider.addEventListener("input", (e) => {
        rangeValEl.innerText = `${e.target.value} px`;
        syncSensorConfig();
    });

    sensorNoiseSwitch.addEventListener("change", () => {
        syncSensorConfig();
    });

    function syncSensorConfig() {
        sendConfig({
            type: "update_config",
            beam_count: parseInt(lidarBeamsSlider.value),
            max_range: parseFloat(lidarRangeSlider.value),
            noise_enabled: sensorNoiseSwitch.checked
        });
    }

    function sendConfig(obj) {
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify(obj));
        }
    }
}

// Helpers
function jsonMsg(type, body) {
    return JSON.stringify({ type, ...body });
}

// Start
connect();
setupInputListeners();
