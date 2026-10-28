document.addEventListener('DOMContentLoaded', function() {
    // Fetch and display accuracy data
    fetchAccuracyData();
    
    // Add hover effects to graph images
    initializeGraphHoverEffects();
    
    // Initialize any other components if needed
    initializeAdditionalFeatures();
});

function fetchAccuracyData() {
    fetch('static/accuracy.json')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            const accuracyElement = document.getElementById('accuracy');
            if (accuracyElement && data.accuracy !== undefined) {
                accuracyElement.textContent = data.accuracy.toFixed(2) + '%';
                accuracyElement.style.opacity = '0';
                
                // Add fade-in animation with delay
                setTimeout(() => {
                    accuracyElement.style.transition = 'opacity 0.8s ease-in-out, transform 0.5s ease';
                    accuracyElement.style.opacity = '1';
                    accuracyElement.style.transform = 'translateY(0)';
                }, 200);
                
                // Add color coding based on accuracy value
                applyAccuracyColor(accuracyElement, data.accuracy);
            } else {
                throw new Error('Accuracy element or data not found');
            }
        })
        .catch(error => {
            console.error('Error fetching accuracy:', error);
            const accuracyElement = document.getElementById('accuracy');
            if (accuracyElement) {
                accuracyElement.textContent = 'N/A';
                accuracyElement.style.color = '#999';
            }
        });
}

function applyAccuracyColor(element, accuracy) {
    // Remove existing color classes
    element.classList.remove('accuracy-high', 'accuracy-medium', 'accuracy-low');
    
    // Apply color based on accuracy range
    if (accuracy >= 90) {
        element.classList.add('accuracy-high');
        element.style.color = '#27ae60'; // Green
    } else if (accuracy >= 80) {
        element.classList.add('accuracy-medium');
        element.style.color = '#f39c12'; // Orange
    } else {
        element.classList.add('accuracy-low');
        element.style.color = '#e74c3c'; // Red
    }
}

function initializeGraphHoverEffects() {
    const graphs = document.querySelectorAll('.graph-img');
    
    graphs.forEach(graph => {
        // Add smooth transition
        graph.style.transition = 'transform 0.3s ease, box-shadow 0.3s ease';
        
        graph.addEventListener('mouseover', () => {
            graph.style.transform = 'scale(1.05)';
            graph.style.boxShadow = '0 8px 25px rgba(0, 0, 0, 0.15)';
            graph.style.zIndex = '10';
        });
        
        graph.addEventListener('mouseout', () => {
            graph.style.transform = 'scale(1)';
            graph.style.boxShadow = '0 2px 10px rgba(0, 0, 0, 0.1)';
            graph.style.zIndex = '1';
        });
        
        // Add click effect for mobile devices
        graph.addEventListener('click', () => {
            graph.style.transform = 'scale(1.03)';
            setTimeout(() => {
                graph.style.transform = 'scale(1.05)';
            }, 150);
        });
    });
}

function initializeAdditionalFeatures() {
    // Add loading states for images
    const images = document.querySelectorAll('img');
    images.forEach(img => {
        img.addEventListener('load', function() {
            this.style.opacity = '1';
        });
        
        img.addEventListener('error', function() {
            console.warn('Image failed to load:', this.src);
            this.style.opacity = '0.5';
        });
        
        // Set initial state
        img.style.transition = 'opacity 0.3s ease';
        img.style.opacity = '0';
    });
    
    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add intersection observer for fade-in animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    // Observe elements with fade-in class
    document.querySelectorAll('.fade-in').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
}

// Utility function to debounce rapid events
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Export functions for global access if needed
window.AppUtils = {
    fetchAccuracyData,
    initializeGraphHoverEffects,
    initializeAdditionalFeatures,
    debounce
};

// Add error boundary for the entire script
window.addEventListener('error', function(e) {
    console.error('Global error caught:', e.error);
});

// Initialize when DOM is fully loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeAll);
} else {
    initializeAll();
}

function initializeAll() {
    // Your existing initialization code here
    fetchAccuracyData();
    initializeGraphHoverEffects();
    initializeAdditionalFeatures();
}