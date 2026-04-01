// Add page-specific CSS classes to body based on URL path
// This lets us target per-page styling without div wrappers
// that interfere with markdown extension processing (e.g. footnotes)
document.addEventListener("DOMContentLoaded", function() {
  var path = window.location.pathname;
  if (path.includes("/paper")) {
    document.body.classList.add("page-paper");
  } else if (path.includes("/constitution")) {
    document.body.classList.add("page-constitution");
  }
});
