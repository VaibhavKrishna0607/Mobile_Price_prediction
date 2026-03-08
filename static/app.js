// Theme Toggle
const themeToggle = document.getElementById('theme-toggle');
const iconSun = themeToggle.querySelector('.icon-sun');
const iconMoon = themeToggle.querySelector('.icon-moon');

// Check for saved theme preference or default to 'dark'
const currentTheme = localStorage.getItem('theme') || 'dark';
document.documentElement.setAttribute('data-theme', currentTheme);

if (currentTheme === 'light') {
    iconSun.style.display = 'none';
    iconMoon.style.display = 'block';
}

themeToggle.addEventListener('click', function() {
    let theme = document.documentElement.getAttribute('data-theme');
    
    if (theme === 'dark') {
        document.documentElement.setAttribute('data-theme', 'light');
        localStorage.setItem('theme', 'light');
        iconSun.style.display = 'none';
        iconMoon.style.display = 'block';
    } else {
        document.documentElement.setAttribute('data-theme', 'dark');
        localStorage.setItem('theme', 'dark');
        iconSun.style.display = 'block';
        iconMoon.style.display = 'none';
    }
});

// Form submission handler
document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const submitBtn = document.getElementById('submitBtn');
    const btnText = submitBtn.querySelector('.btn-text');
    const btnLoader = submitBtn.querySelector('.btn-loader');
    const resultCard = document.getElementById('resultCard');
    const errorCard = document.getElementById('errorCard');
    const placeholderCard = document.getElementById('placeholderCard');
    
    // Hide previous results
    resultCard.style.display = 'none';
    errorCard.style.display = 'none';
    placeholderCard.style.display = 'none';
    
    // Show loading state
    submitBtn.disabled = true;
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline';
    
    // Collect form data
    const formData = new FormData(e.target);
    const data = {};
    
    // Convert FormData to object
    for (const [key, value] of formData.entries()) {
        data[key] = value;
    }
    
    try {
        // Make prediction request
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Display success result
            displayResult(result);
        } else {
            // Display error
            displayError(result.error || 'Prediction failed. Please try again.');
        }
    } catch (error) {
        console.error('Error:', error);
        displayError('Network error. Please check your connection and try again.');
    } finally {
        // Reset button state
        submitBtn.disabled = false;
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
});

function displayResult(result) {
    const resultCard = document.getElementById('resultCard');
    const priceRangeElement = document.getElementById('priceRange');
    const predictionDetailsElement = document.getElementById('predictionDetails');
    
    // Map prediction to class and display text
    const priceRanges = {
        0: { text: 'Budget Phone', class: 'budget', price: '₹0 - ₹10,000' },
        1: { text: 'Lower Mid-Range', class: 'lower-mid', price: '₹10,000 - ₹20,000' },
        2: { text: 'Upper Mid-Range', class: 'upper-mid', price: '₹20,000 - ₹35,000' },
        3: { text: 'Premium Phone', class: 'premium', price: '₹35,000+' }
    };
    
    const rangeInfo = priceRanges[result.prediction] || priceRanges[0];
    
    // Update price range display
    priceRangeElement.textContent = rangeInfo.text;
    priceRangeElement.className = `price-range ${rangeInfo.class}`;
    
    // Update prediction details
    predictionDetailsElement.innerHTML = `
        <h3>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="18" y1="20" x2="18" y2="10"/>
                <line x1="12" y1="20" x2="12" y2="4"/>
                <line x1="6" y1="20" x2="6" y2="14"/>
            </svg>
            Details
        </h3>
        <p><strong>Price Range:</strong> ${rangeInfo.price}</p>
        <p><strong>Category:</strong> ${rangeInfo.text}</p>
        <p><strong>Confidence:</strong> Based on ML model</p>
    `;
    
    // Show result card
    resultCard.style.display = 'block';
    
    // Show recommendations if available
    if (result.recommended_phones && result.recommended_phones.length > 0) {
        displayRecommendations(result.recommended_phones);
    }
}

function displayRecommendations(phones) {
    const container = document.getElementById('recommendationsCard');
    const grid = document.getElementById('phoneGrid');

    grid.innerHTML = phones.map(phone => `
        <div class="phone-card">
            <div class="phone-card-header">
                <span class="phone-brand">${phone.brand_name}</span>
                ${phone['5G_or_not'] ? '<span class="badge-5g">5G</span>' : ''}
            </div>
            <div class="phone-name">${phone.model}</div>
            <div class="phone-price">₹${Number(phone.price).toLocaleString('en-IN')}</div>
            <div class="phone-rating">${renderStars(phone.avg_rating)} <span>${Number(phone.avg_rating).toFixed(1)}</span></div>
            <div class="phone-specs">
                <div class="spec-item">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="4" y="2" width="16" height="20" rx="2" ry="2"/>
                        <line x1="12" y1="18" x2="12.01" y2="18"/>
                    </svg>
                    ${phone.ram_capacity} GB RAM
                </div>
                <div class="spec-item">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="1" y="6" width="18" height="12" rx="2" ry="2"/>
                        <path d="M23 13v-2"/>
                    </svg>
                    ${phone.battery_capacity} mAh
                </div>
                <div class="spec-item">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
                        <circle cx="12" cy="13" r="4"/>
                    </svg>
                    ${phone.primary_camera_rear} MP
                </div>
                <div class="spec-item">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="2" y="7" width="20" height="14" rx="2" ry="2"/>
                        <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>
                    </svg>
                    ${phone.internal_memory} GB
                </div>
                <div class="spec-item">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M23 6l-9.5 9.5-5-5L1 18"/>
                    </svg>
                    ${phone.refresh_rate} Hz
                </div>
            </div>
        </div>
    `).join('');

    container.style.display = 'block';
}

function renderStars(rating) {
    const full = Math.round(rating / 2); // rating is out of 10, stars out of 5
    return Array.from({length: 5}, (_, i) =>
        `<span class="star ${i < full ? 'filled' : ''}">★</span>`
    ).join('');
}

function displayError(errorMessage) {
    const errorCard = document.getElementById('errorCard');
    const errorMessageElement = document.getElementById('errorMessage');
    
    errorMessageElement.textContent = errorMessage;
    errorCard.style.display = 'block';
}

function resetForm() {
    // Reset form
    document.getElementById('predictionForm').reset();
    
    // Hide results
    document.getElementById('resultCard').style.display = 'none';
    document.getElementById('errorCard').style.display = 'none';
    document.getElementById('recommendationsCard').style.display = 'none';
    
    // Show placeholder
    document.getElementById('placeholderCard').style.display = 'flex';
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Add input validation
document.querySelectorAll('input[type="number"]').forEach(input => {
    input.addEventListener('input', function() {
        if (this.value === '') return;
        
        const min = parseFloat(this.min);
        const max = parseFloat(this.max);
        const value = parseFloat(this.value);
        
        if (isNaN(value)) {
            this.setCustomValidity('Please enter a valid number');
        } else if (value < min) {
            this.setCustomValidity(`Value must be at least ${min}`);
        } else if (value > max) {
            this.setCustomValidity(`Value must be at most ${max}`);
        } else {
            this.setCustomValidity('');
        }
    });
    
    input.addEventListener('blur', function() {
        if (this.value !== '' && !this.checkValidity()) {
            this.reportValidity();
        }
    });
});

// Random value generator with realistic ranges
function getRandomFromArray(arr) {
    return arr[Math.floor(Math.random() * arr.length)];
}

// Sample data function with random values for dropdowns
window.fillSampleData = function() {
    const selectFields = {
        ram_capacity:              ['2', '3', '4', '6', '8', '12', '16'],
        internal_memory:           ['32', '64', '128', '256', '512'],
        processor_speed:           ['1.8', '2.0', '2.2', '2.4', '2.6', '2.8', '3.0'],
        num_cores:                 ['4', '6', '8'],
        primary_camera_rear:       ['12', '48', '50', '64', '108', '200'],
        primary_camera_front:      ['8', '16', '32', '50'],
        num_rear_cameras:          ['1', '2', '3', '4'],
        resolution_height:         ['720', '1080', '1600', '2400', '2772'],
        resolution_width:          ['720', '1080', '1440'],
        screen_size:               ['5.5', '6.0', '6.5', '6.7', '7.0'],
        refresh_rate:              ['60', '90', '120', '144'],
        battery_capacity:          ['3000', '4000', '4500', '5000', '5500', '6000'],
        five_g:                    ['0', '1'],
        fast_charging_available:   ['0', '1'],
        extended_memory_available: ['0', '1'],
    };

    Object.keys(selectFields).forEach(fieldId => {
        const select = document.getElementById(fieldId);
        if (select) {
            select.value = getRandomFromArray(selectFields[fieldId]);
            select.setCustomValidity('');
            select.dispatchEvent(new Event('change', { bubbles: true }));
        }
    });

    console.log('Random device specs selected!');
};

console.log('Price Predictor ready! Use "Quick Fill" to populate sample data.');
