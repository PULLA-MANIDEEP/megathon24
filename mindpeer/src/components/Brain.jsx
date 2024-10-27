import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { MeshDistortMaterial, Sphere } from '@react-three/drei';

function Brain() {
  const brainRef = useRef();

  // Rotate the brain model slightly based on mouse movement
  useFrame((state) => {
    const { mouse } = state;
    if (brainRef.current) {
      brainRef.current.rotation.y = mouse.x * Math.PI; // Horizontal rotation
      brainRef.current.rotation.x = -mouse.y * Math.PI; // Vertical rotation
    }
  });

  return (
    <Sphere ref={brainRef} args={[1, 64, 64]} scale={2.5}>
      <MeshDistortMaterial color="purple" attach="material" distort={0.3} speed={3} />
    </Sphere>
  );
}

export default Brain;
