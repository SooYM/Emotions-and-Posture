const fs = require('fs');
const path = require('path');

const datasetDir = path.join(__dirname, 'Dataset');
const outputFile = path.join(__dirname, 'public', 'dataset_manifest.json');

function walkDir(dir, fileList = []) {
  const files = fs.readdirSync(dir);
  for (const file of files) {
    if (file.startsWith('.')) continue; // skip hidden files
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);
    if (stat.isDirectory()) {
      walkDir(filePath, fileList);
    } else if (/\.(jpe?g|png|webp)$/i.test(file)) {
      // Get path relative to the project root
      const relativePath = path.relative(__dirname, filePath);
      fileList.push(relativePath);
    }
  }
  return fileList;
}

try {
  if (fs.existsSync(datasetDir)) {
    const list = walkDir(datasetDir);
    // Ensure public folder exists
    if (!fs.existsSync(path.join(__dirname, 'public'))) {
      fs.mkdirSync(path.join(__dirname, 'public'));
    }
    fs.writeFileSync(outputFile, JSON.stringify(list, null, 2));
    console.log(`Successfully generated manifest with ${list.length} images.`);
  } else {
    console.error("Dataset folder not found!");
  }
} catch (error) {
  console.error("Error generating manifest:", error);
}
