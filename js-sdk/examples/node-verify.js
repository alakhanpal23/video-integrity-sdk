#!/usr/bin/env node
/**
 * Example: Node.js verification using the JS SDK
 */

const { verifyVideo } = require('../index');
const fs = require('fs');
const path = require('path');

async function createTestVideo(outputPath) {
    // This would normally use ffmpeg or similar
    // For demo purposes, we'll create a minimal MP4 file
    const minimalMp4 = Buffer.from([
        0x00, 0x00, 0x00, 0x20, 0x66, 0x74, 0x79, 0x70, // ftyp box
        0x6d, 0x70, 0x34, 0x31, 0x00, 0x00, 0x00, 0x00,
        0x6d, 0x70, 0x34, 0x31, 0x69, 0x73, 0x6f, 0x6d,
        // Add some padding to make it look like a real file
        ...Array(1000).fill(0)
    ]);
    
    fs.writeFileSync(outputPath, minimalMp4);
    console.log(`✅ Created test video: ${outputPath}`);
}

async function main() {
    const testVideoPath = path.join(__dirname, 'test_video.mp4');
    const secret = 'ffeeddccbbaa99887766554433221100';
    const apiUrl = process.env.API_URL || 'http://127.0.0.1:8000/verify';
    
    try {
        // Create test video
        console.log('📹 Creating test video...');
        await createTestVideo(testVideoPath);
        
        // Set API key if available
        if (process.env.VIDEO_SDK_API_KEY) {
            console.log(`🔐 Using API key: ${process.env.VIDEO_SDK_API_KEY.substring(0, 8)}...`);
        }
        
        // Verify video
        console.log('🔍 Verifying video via JS SDK...');
        
        const result = await verifyVideo(
            testVideoPath,
            secret,
            8,  // n_frames
            apiUrl
        );
        
        console.log('✅ Verification completed!');
        console.log(`   Valid: ${result.valid}`);
        console.log(`   BER: ${result.ber.toFixed(4)}`);
        console.log(`   Summary: ${result.summary}`);
        
        if (result.processing_time) {
            console.log(`   Processing time: ${result.processing_time.toFixed(2)}s`);
        }
        
        // Test error handling
        console.log('\\n🧪 Testing error handling...');
        try {
            await verifyVideo('nonexistent.mp4', secret);
        } catch (error) {
            console.log(`✅ Error handling works: ${error.message}`);
        }
        
    } catch (error) {
        console.error('❌ Error:', error.message);
        
        if (error.message.includes('ECONNREFUSED')) {
            console.log('\\n💡 Make sure the API server is running:');
            console.log('   uvicorn src.api:app --host 0.0.0.0 --port 8000');
        }
        
        if (error.response?.status === 401) {
            console.log('\\n💡 Set your API key:');
            console.log('   export VIDEO_SDK_API_KEY=your-api-key');
        }
        
        process.exit(1);
        
    } finally {
        // Cleanup
        if (fs.existsSync(testVideoPath)) {
            fs.unlinkSync(testVideoPath);
        }
    }
}

// Handle command line usage
if (require.main === module) {
    main().catch(console.error);
}

module.exports = { main };