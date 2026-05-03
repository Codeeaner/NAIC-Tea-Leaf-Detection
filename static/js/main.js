window.showAlert = showAlert;
window.formatFileSize = formatFileSize;
window.formatDate = formatDate;

function showAlert(message, type = 'info') {
    let backgroundColor;
    switch (type) {
        case 'success':
            backgroundColor = 'linear-gradient(to right, #00b09b, #96c93d)';
            break;
        case 'warning':
            backgroundColor = 'linear-gradient(to right, #ffc107, #ff822e)';
            break;
        case 'danger':
            backgroundColor = 'linear-gradient(to right, #ff5f6d, #ffc371)';
            break;
        default:
            backgroundColor = 'linear-gradient(to right, #0083B0, #00B4DB)';
    }

    Toastify({
        text: message,
        duration: 3000,
        close: true,
        gravity: "bottom", // `top` or `bottom`
        position: "right", // `left`, `center` or `right`
        backgroundColor: backgroundColor,
        stopOnFocus: true, // Prevents dismissing of toast on hover
        className: "toastify", // Add this line to apply custom styles
    }).showToast();
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString();
}
