let fileData = null;
let fileContents = {};


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
        <div class="result-item" onclick="openFile('data/${result.path}')">
            <div class="file-preview">
                ${getFilePreview(result)}
            </div>
            <div class="result-info">
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
            </div>
        </div>
    `).join('');
}

function openFile(path) {
    window.open(path, '_blank');
}

function getFilePreview(result) {
    if (result.type === 'image') {
        return `<img src="data/${result.path}" alt="${result.path}">`;
    } else if (result.type === 'text') {
        const content = fileContents[result.path] || 'Loading...';
        return `<div class="text-content">${escapeHtml(content)}</div>`;
    }
    return '';
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

loadData();
