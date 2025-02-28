let fileData = null;
let fileContents = {};
let currentModel = null;

// Color mapping for string colors
const colorMap = {
    'red': '#FF0000',
    'green': '#008000',
    'blue': '#0000FF',
    'black': '#000000',
    'white': '#FFFFFF',
    'yellow': '#FFFF00',
    'purple': '#800080',
    'orange': '#FFA500',
    'brown': '#A52A2A',
    'pink': '#FFC0CB',
    'gray': '#808080',
    'unknown': '#CCCCCC'
};

async function loadData() {
    try {
        const response = await fetch('data/tags.json');
        fileData = await response.json();
        console.log('Data loaded:', fileData);
        // Pre-load text contents
        await loadTextContents();
    } catch (error) {
        console.error('Error loading JSON:', error);
    }
}

async function loadTextContents() {
    for (const [path, file] of Object.entries(fileData.files)) {
        if (file.type === 'text') {
            try {
                const response = await fetch('data/' + path);
                fileContents[path] = await response.text();
            } catch (error) {
                console.error(`Error loading text file ${path}:`, error);
                fileContents[path] = 'Error loading file content';
            }
        }
    }
}

function searchFiles(query) {
    if (!fileData || !query) return [];
    
    query = query.toLowerCase();
    const searchTerms = query.split(/\s+/); // Split on whitespace
    const results = [];
    
    for (const [path, file] of Object.entries(fileData.files)) {
        // Check if all search terms match
        const hasAllTerms = searchTerms.every(term => {
            return file.tags.some(tagObj => {
                const tagText = tagObj.tag.toLowerCase();
                // Check if the tag contains the term or if multiple tags together match the term
                return tagText.includes(term) || 
                       file.tags.map(t => t.tag.toLowerCase()).join(' ').includes(term);
            });
        });
        
        if (hasAllTerms) {
            results.push({
                path,
                ...file
            });
        }
    }
    
    return results;
}

function displayResults(results) {
    const resultsContainer = document.getElementById('results');
    
    if (results.length === 0) {
        resultsContainer.style.display = 'none';
        return;
    }
    
    resultsContainer.style.display = 'grid';
    resultsContainer.innerHTML = results.map(result => `
        <div class="result-item">
            <div class="file-preview">
                ${getFilePreview(result)}
            </div>
            <div class="result-info">
                <div class="file-type">${result.type}</div>
                <div class="file-path">${result.path}</div>
                ${result.type === 'image' ? `
                    <div class="color-indicator">
                        <span class="color-dot" style="background-color: ${colorMap[result.color] || result.color}"></span>
                        <span>${result.color}</span>
                    </div>
                ` : ''}
                <div class="tags">
                    ${result.tags.map(tag => 
                        `<span class="tag">${tag.tag} (${tag.confidence})</span>`
                    ).join('')}
                </div>
                ${result.type === '3d' ? `
                    <button class="view-3d-btn" data-path="${result.path}">View 3D Model</button>
                ` : ''}
            </div>
        </div>
    `).join('');
    
    // Add event listeners for 3D model buttons
    document.querySelectorAll('.view-3d-btn').forEach(button => {
        button.addEventListener('click', function() {
            const modelPath = this.getAttribute('data-path');
            openModelViewer(modelPath);
        });
    });
}



function getFilePreview(result) {
    if (result.type === 'image') {
        return `<img src="data/${result.path}" alt="${result.path}">`;
    } else if (result.type === 'text') {
        const content = fileContents[result.path] || 'Loading...';
        return `<div class="text-content">${escapeHtml(content)}</div>`;
    } else if (result.type === '3d') {
        // If there's a thumbnail, use it without the overlay
        if (result.thumbnail) {
            return `<div class="model-placeholder">
                        <img src="data/${result.thumbnail}" alt="${result.path}" class="model-thumbnail">
                    </div>`;
        } else {
            // Otherwise, use the generic placeholder
            return `<div class="model-placeholder">
                        <div class="model-icon">
                            <svg width="64" height="64" viewBox="0 0 24 24">
                                <path fill="#666" d="M12,0L3,7L4,8.18V16.18L12,21L20,16.18V8.18L21,7L12,0M12,2.5L17.5,6.5L12,10.5L6.5,6.5L12,2.5M5,9.21V15.07L11,19V13.35L5,9.21M19,9.21L13,13.35V19L19,15.07V9.21Z"/>
                            </svg>
                        </div>
                        <div>3D Model Preview</div>
                    </div>`;
        }
    }
    return '';
}





function openModelViewer(modelPath) {
    // Create modal if it doesn't exist
    let modal = document.getElementById('model-viewer-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'model-viewer-modal';
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content">
                <span class="close-modal">&times;</span>
                <div id="model-container"></div>
            </div>
        `;
        document.body.appendChild(modal);
        
        // Add close button event
        modal.querySelector('.close-modal').addEventListener('click', closeModelViewer);
    }
    
    // Show modal
    modal.style.display = 'block';
    
    // Initialize Three.js viewer
    initThreeJsViewer('data/' + modelPath);
}

function closeModelViewer() {
    const modal = document.getElementById('model-viewer-modal');
    if (modal) {
        modal.style.display = 'none';
    }
    
    // Clean up Three.js resources
    if (currentModel) {
        currentModel.dispose();
        currentModel = null;
    }
}


function initThreeJsViewer(modelUrl) {
    // Get container
    const container = document.getElementById('model-container');
    container.innerHTML = '';
    
    // Set up scene
    const scene = new THREE.Scene();
    // Remove the background color
    // scene.background = new THREE.Color(0xf0f0f0);
    
    // Camera
    const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.z = 5;
    
    // Renderer with alpha
    const renderer = new THREE.WebGLRenderer({ 
        antialias: true,
        alpha: true  // Enable transparency
    });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setClearColor(0x000000, 0); // Set clear color with 0 alpha (transparent)
    container.appendChild(renderer.domElement);
        
    // Lights
    const ambientLight = new THREE.AmbientLight(0xcccccc, 0.4);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);
    
    // Controls
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.25;
    
    // Loading indicator
    const loadingText = document.createElement('div');
    loadingText.style.position = 'absolute';
    loadingText.style.top = '50%';
    loadingText.style.width = '100%';
    loadingText.style.textAlign = 'center';
    loadingText.style.color = '#333';
    loadingText.innerHTML = 'Loading 3D model...';
    container.appendChild(loadingText);
    
    // Simplified loader usage based on the HTML imports
    const loader = new THREE.GLTFLoader();
    loader.load(
        modelUrl,
        function (gltf) {
            // Success callback
            container.removeChild(loadingText);
            scene.add(gltf.scene);
            currentModel = gltf.scene;
            
            // Make materials emissive
            gltf.scene.traverse(function(node) {
                if (node.isMesh && node.material) {
                    // Handle both single material and array of materials
                    if (Array.isArray(node.material)) {
                        node.material.forEach(material => {
                            // Make sure material is not null
                            if (material) {
                                material.emissive = new THREE.Color(0x404040);
                                material.emissiveIntensity = 0.3;
                            }
                        });
                    } else if (node.material) {
                        node.material.emissive = new THREE.Color(0x404040);
                        node.material.emissiveIntensity = 0.3;
                    }
                }
            });
            
            // Center and scale model
            const box = new THREE.Box3().setFromObject(gltf.scene);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());
            
            gltf.scene.position.x = -center.x;
            gltf.scene.position.y = -center.y;
            gltf.scene.position.z = -center.z;
            
            const maxDim = Math.max(size.x, size.y, size.z);
            const fov = camera.fov * (Math.PI / 180);
            let cameraZ = Math.abs(maxDim / Math.sin(fov / 2));
            camera.position.z = cameraZ * 1.5;
            
            const minZ = box.min.z;
            const cameraToFarEdge = (minZ < 0) ? -minZ + cameraZ : cameraZ - minZ;
            camera.far = cameraToFarEdge * 3;
            camera.updateProjectionMatrix();
            
            controls.maxDistance = cameraToFarEdge * 2;
            controls.target = center;
            controls.update();
        },
        function (xhr) {
            // Progress callback
            const percent = (xhr.loaded / xhr.total) * 100;
            loadingText.innerHTML = `Loading: ${Math.round(percent)}%`;
        },
        function (error) {
            // Error callback
            console.error('Error loading model:', error);
            loadingText.innerHTML = 'Error loading model';
        }
    );
    
    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    animate();
    
    // Handle window resize
    window.addEventListener('resize', function() {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    });
}












function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

document.getElementById('searchInput').addEventListener('input', (e) => {
    const query = e.target.value.trim();
    const results = searchFiles(query);
    displayResults(results);
});

// Load data when page loads
loadData();
