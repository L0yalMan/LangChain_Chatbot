import React from 'react';

function LoadingIndicator() {
  return (
    <div className="loading-indicator text-center p-4 text-indigo-700">
      <div className="spinner border-4 border-gray-200 border-l-indigo-700 rounded-full w-6 h-6 animate-spin mx-auto mb-2"></div>
      <span>AI is typing...</span>
    </div>
  );
}

export default LoadingIndicator; 