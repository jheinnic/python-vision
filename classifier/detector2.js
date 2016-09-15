cv = require("opencv");

var color = [0, 255, 0];
var thickness = 2;

var cascadeFile = "classifiers/all/cascade.xml";

var inputFiles = [ "../train5/sample01.tif", "../train5/sample02.tif" ]

inputFiles.forEach(function(fileName) {
  // console.log(fileName);
  cv.readImage(fileName, function(err, im) {
    im.detectObject(cascadeFile, {neighbors: 0, scale: 1.1}, function(err, objects) {
      /*if(err != undefined) {
         console.log(err);
         return;
      }*/
      console.log(objects);
      for(var k = 0; k < objects.length; k++) {
        var object = objects[k];
        im.rectangle([object.x, object.y], [object.width, object.height], color, 2);
      }
      im.save(fileName.replace(/.tif/, "processed.tif").replace('../train5', '.'));
    });
  });
});
