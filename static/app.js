// Form submission handler
document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const submitBtn = document.getElementById('submitBtn');
    const btnText = submitBtn.querySelector('.btn-text');
    const btnLoader = submitBtn.querySelector('.btn-loader');
    const resultContainer = document.getElementById('resultContainer');
    const errorContainer = document.getElementById('errorContainer');
    
    // Hide previous results
    resultContainer.style.display = 'none';
    errorContainer.style.display = 'none';
    
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
    const resultContainer = document.getElementById('resultContainer');
    const priceRangeElement = document.getElementById('priceRange');
    const predictionDetailsElement = document.getElementById('predictionDetails');
    
    // Map prediction to class and display text
    const priceRanges = {
        0: { text: 'Budget mobile phone', class: 'budget', price: '₹0 - ₹10,000' },
        1: { text: 'Lower mid-range phone', class: 'lower-mid', price: '₹10,000 - ₹20,000' },
        2: { text: 'Upper mid-range phone', class: 'upper-mid', price: '₹20,000 - ₹35,000' },
        3: { text: 'Premium phone', class: 'premium', price: '₹35,000+' }
    };
    
    const rangeInfo = priceRanges[result.prediction] || priceRanges[0];
    
    // Update price range display
    priceRangeElement.textContent = rangeInfo.text;
    priceRangeElement.className = `price-range ${rangeInfo.class}`;
    
    // Update prediction details
    predictionDetailsElement.innerHTML = `
        <h3>📊 Prediction Details</h3>
        <p><strong>Price Range:</strong> ${rangeInfo.price}</p>
        <p><strong>Category:</strong> ${rangeInfo.text}</p>
        <p><strong>Prediction Confidence:</strong> Based on your specifications</p>
    `;
    
    // Show result container
    resultContainer.style.display = 'block';
    
    // Scroll to result
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function displayError(errorMessage) {
    const errorContainer = document.getElementById('errorContainer');
    const errorMessageElement = document.getElementById('errorMessage');
    
    errorMessageElement.textContent = errorMessage;
    errorContainer.style.display = 'block';
    
    // Scroll to error
    errorContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function resetForm() {
    // Reset form
    document.getElementById('predictionForm').reset();
    
    // Hide results
    document.getElementById('resultContainer').style.display = 'none';
    document.getElementById('errorContainer').style.display = 'none';
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Add input validation and helpful tooltips
document.querySelectorAll('input[type="number"]').forEach(input => {
    input.addEventListener('input', function() {
        const min = parseFloat(this.min);
        const max = parseFloat(this.max);
        const value = parseFloat(this.value);
        
        if (value < min) {
            this.setCustomValidity(`Value must be at least ${min}`);
        } else if (value > max) {
            this.setCustomValidity(`Value must be at most ${max}`);
        } else {
            this.setCustomValidity('');
        }
    });
    
    // Show validation on blur
    input.addEventListener('blur', function() {
        if (!this.checkValidity()) {
            this.reportValidity();
        }
    });
});

// Add sample data button (for testing)
function fillSampleData() {
    // Sample smartphone specifications
    const sampleData = {
        battery_power: 2000,
        ram: 3072, // 3GB
        int_memory: 32,
        clock_speed: 2.0,
        n_cores: 4,
        pc: 12,
        fc: 5,
        px_height: 1960,
        px_width: 1080,
        sc_h: 12.5,
        sc_w: 6.2,
        touch_screen: 1,
        mobile_wt: 180,
        m_dep: 0.8,
        talk_time: 10,
        three_g: 1,
        four_g: 1,
        wifi: 1,
        blue: 1,
        dual_sim: 1
    };
    
    // Fill form with sample data
    Object.keys(sampleData).forEach(key => {
        const input = document.getElementById(key);
        if (input) {
            input.value = sampleData[key];
        }
    });
}

// Uncomment the line below to add a "Fill Sample Data" button for testing
// console.log('To test with sample data, call fillSampleData() in the console');

