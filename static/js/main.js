// Format a date string to a readable Spanish format
function formatDate(dateString) {
    const options = {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    };
    return new Date(dateString).toLocaleDateString('es-ES', options);
}

// Show a notification message on the page
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed top-0 end-0 m-3`;
    notification.style.zIndex = '1000';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    document.body.appendChild(notification);

    // Auto-close after 5 seconds
    setTimeout(() => {
        notification.remove();
    }, 5000);
}

// Update the last detection section with animation
async function updateLastDetection() {
    try {
        const response = await fetch('/last-detection/');
        const data = await response.json();
        const detectionElement = document.getElementById('lastDetection');

        if (data.message) {
            // No detection found
            detectionElement.innerHTML = `
                <p class="lead">${data.message}</p>
            `;
        } else {
            // Add animation class
            detectionElement.classList.add('fade-in');

            detectionElement.innerHTML = `
                <p class="lead">Tipo: ${data.waste_type}</p>
                <p>Confianza: ${(data.confidence * 100).toFixed(1)}%</p>
                <p>Fecha: ${formatDate(data.detection_date)}</p>
            `;

            // Remove animation class after animation
            setTimeout(() => {
                detectionElement.classList.remove('fade-in');
            }, 1000);
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification('Error updating last detection', 'danger');
    }
}

// Update the status of all bins with animation and notifications
async function updateBinStatus() {
    try {
        const response = await fetch('/bins-status/');
        const data = await response.json();

        data.bins.forEach(bin => {
            const statusElement = document.getElementById(`${bin.bin_type.toLowerCase()}Status`);
            if (statusElement) {
                // Show 'Full' or 'Available' depending on bin status
                const newStatus = bin.is_full ? 'Llena' : 'Disponible';
                const newClass = `bin-status ${bin.is_full ? 'full' : 'empty'}`;

                // Only update if the status has changed
                if (statusElement.textContent !== newStatus) {
                    statusElement.textContent = newStatus;
                    statusElement.className = newClass;

                    // Show notification when bin status changes
                    showNotification(`La caneca de ${bin.bin_type} está ${newStatus.toLowerCase()}`,
                        bin.is_full ? 'warning' : 'success');
                }
            }
        });
    } catch (error) {
        console.error('Error:', error);
        showNotification('Error updating bin status', 'danger');
    }
}

// Update all data on the page (last detection and bin status)
function updateData() {
    updateLastDetection();
    updateBinStatus();
}

// Initialize the app: update data and set interval for updates
document.addEventListener('DOMContentLoaded', () => {
    // Initial data update
    updateData();

    // Update data every 30 seconds
    setInterval(updateData, 30000);

    // Show welcome message
    showNotification('Welcome to the STICY System', 'info');
}); 