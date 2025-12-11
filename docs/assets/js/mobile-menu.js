document.addEventListener('DOMContentLoaded', function() {
  // Mobile main navigation toggle
  const mobileMenuToggle = document.querySelector('.mobile-menu-toggle');
  const mainNav = document.querySelector('.main-nav');
  
  if (mobileMenuToggle && mainNav) {
    mobileMenuToggle.addEventListener('click', function(e) {
      e.preventDefault();
      e.stopPropagation();
      this.classList.toggle('active');
      mainNav.classList.toggle('active');
      document.body.classList.toggle('menu-open');
    });
  }
  
  // Sidebar section toggles for mobile
  const sidebarHeadings = document.querySelectorAll('.sidebar-nav h3');
  
  sidebarHeadings.forEach(heading => {
    heading.addEventListener('click', function() {
      if (window.innerWidth <= 768) {
        const nextUl = this.nextElementSibling;
        if (nextUl && nextUl.tagName === 'UL') {
          nextUl.classList.toggle('expanded');
          this.classList.toggle('expanded');
        }
      }
    });
  });
  
  // Close mobile menu when clicking outside
  document.addEventListener('click', function(event) {
    if (
      mainNav && 
      mainNav.classList.contains('active') && 
      !mainNav.contains(event.target) && 
      !mobileMenuToggle.contains(event.target)
    ) {
      mainNav.classList.remove('active');
      mobileMenuToggle.classList.remove('active');
      document.body.classList.remove('menu-open');
    }
  });
  
  // Close mobile menu when window is resized to desktop size
  window.addEventListener('resize', function() {
    if (mainNav && window.innerWidth > 768) {
      mainNav.classList.remove('active');
      if (mobileMenuToggle) {
        mobileMenuToggle.classList.remove('active');
      }
      document.body.classList.remove('menu-open');
      
      // Expand all sidebar sections on desktop
      document.querySelectorAll('.sidebar-nav ul').forEach(ul => {
        ul.classList.add('expanded');
      });
    }
  });
  
  // Initially expand all sidebar sections on desktop
  if (window.innerWidth > 768) {
    document.querySelectorAll('.sidebar-nav ul').forEach(ul => {
      ul.classList.add('expanded');
    });
  }
});