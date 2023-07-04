//javascript code

var container = document.getElementById('viewer');
var renderer, scene, camera, model;

function init() {
    // Setup renderer
    renderer = new THREE.WebGLRenderer();
    renderer.setSize(container.offsetWidth, container.offsetHeight);
    container.appendChild(renderer.domElement);

    // Setup scene
    scene = new THREE.Scene();

    // Setup camera
    camera = new THREE.PerspectiveCamera(45, container.offsetWidth / container.offsetHeight, 1, 1000);
    camera.position.set(0, 0, 5);
    scene.add(camera);

    // Load STL model
    var loader = new THREE.STLLoader();
    loader.load('path/to/your-model.stl', function(geometry) {
        var material = new THREE.MeshPhongMaterial({ color: 0x00ff00 }); // Set desired material properties
        model = new THREE.Mesh(geometry, material);
        scene.add(model);
    });

    // Add lights
    var ambientLight = new THREE.AmbientLight(0x404040);
    scene.add(ambientLight);
    var directionalLight = new THREE.DirectionalLight(0xffffff);
    directionalLight.position.set(1, 1, 1).normalize();
    scene.add(directionalLight);

    // Render loop
    function animate() {
        requestAnimationFrame(animate);
        renderer.render(scene, camera);
    }
    animate();
}

init();
