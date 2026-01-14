import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Importing the classes to be tested.
# In a real scenario, these would be imported from the source file.
# Here we assume the code provided is available in the scope.
# We will mock the environment for the test.

class TestVizuNet(unittest.TestCase):
    def setUp(self):
        # Initialize model and visualizer on CPU for testing
        self.model = VizuNet()
        self.visualizer = DeconvNetVisualizer(self.model)
        
        # Create a dummy input tensor: Batch Size 2, 3 Channels, 32x32 Image
        self.dummy_input = torch.randn(2, 3, 32, 32)

    def test_forward_pass_shapes(self):
        """Verify that the model processes input and stores indices correctly."""
        output = self.model(self.dummy_input)
        
        # Check Output Shape (Batch, Classes)
        self.assertEqual(output.shape, (2, 10))
        
        # Check Internal Storage
        self.assertIn('layer1', self.model.feature_maps)
        self.assertIn('layer2', self.model.feature_maps)
        self.assertIn('pool1', self.model.switches)
        self.assertIn('pool2', self.model.switches)
        
        # Check Shape Logic
        # Layer 1: Conv(32x32) -> 32x32. Pool(2x2) -> 16x16
        self.assertEqual(self.model.feature_maps['layer1'].shape, (2, 32, 32, 32))
        self.assertEqual(self.model.switches['pool1'].shape, (2, 32, 16, 16))
        
        # Layer 2: Conv(16x16) -> 16x16. Pool(2x2) -> 8x8
        self.assertEqual(self.model.feature_maps['layer2'].shape, (2, 64, 16, 16))
        self.assertEqual(self.model.switches['pool2'].shape, (2, 64, 8, 8))

    def test_reconstruction_layer1(self):
        """Test DeconvNet logic for the first layer."""
        # 1. Reconstruct requires a forward pass first to populate indices
        _ = self.model(self.dummy_input)
        
        # 2. Reconstruct Feature 0 from Layer 1
        recon = self.visualizer.reconstruct(self.dummy_input, 'layer1', feature_idx=0)
        
        # 3. Output should match input image dimensions
        self.assertEqual(recon.shape, (2, 3, 32, 32))
        
    def test_reconstruction_layer2(self):
        """Test DeconvNet logic for the second layer (includes unpooling)."""
        _ = self.model(self.dummy_input)
        
        # Reconstruct Feature 0 from Layer 2
        recon = self.visualizer.reconstruct(self.dummy_input, 'layer2', feature_idx=0)
        
        # Should traverse: Unconv2 -> Unpool1 -> Unconv1 -> Image
        self.assertEqual(recon.shape, (2, 3, 32, 32))

    def test_reconstruction_determinism(self):
        """Ensure reconstruction is deterministic for the same input."""
        _ = self.model(self.dummy_input)
        recon1 = self.visualizer.reconstruct(self.dummy_input, 'layer2', feature_idx=5)
        recon2 = self.visualizer.reconstruct(self.dummy_input, 'layer2', feature_idx=5)
        self.assertTrue(torch.allclose(recon1, recon2))

    def test_invalid_layer_name(self):
        """Ensure graceful handling or None return for invalid layers."""
        _ = self.model(self.dummy_input)
        recon = self.visualizer.reconstruct(self.dummy_input, 'layer_invalid', 0)
        self.assertIsNone(recon)

    def test_feature_isolation(self):
        """Verify that the mask logic isolates the requested feature."""
        # We manually verify the mask logic works by inspecting the visualizer code logic via a mock or derived check.
        # Alternatively, we check that different feature indices produce different reconstructions.
        _ = self.model(self.dummy_input)
        recon_a = self.visualizer.reconstruct(self.dummy_input, 'layer2', feature_idx=0)
        recon_b = self.visualizer.reconstruct(self.dummy_input, 'layer2', feature_idx=1)
        
        # Unless the model weights are zero or symmetric, these should differ
        # Initialize weights to something non-zero/random (default init does this)
        self.assertFalse(torch.allclose(recon_a, recon_b))

if __name__ == '__main__':
    unittest.main()