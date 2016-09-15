cv = require("opencv");

var color = [0, 255, 0];
var thickness = 2;

var cascadeFile = "classifiers/E/cascade.xml";

var inputFiles = [ "test/B01.png", "test/B04.png", "test/C02.png", "test/C04.png", "test/C05.png", "test/C06.png", "test/C07.png", "test/C08.png", "test/D03.png", "test/E02.png", "test/E04.png", "test/E05.png", "test/G01.png", "test/G02.png", "test/G06.png", "test/G07.png", "test/G08.png", "test/H02.png", "test/H05.png", "test/H06.png", "test/I02.png", "test/I03.png", "test/I04.png", "test/J01.png", "test/K01.png", "test/L01.png", "test/L02.png", "test/L03.png", "test/M01.png", "test/M02.png", "test/M03.png", "test/N03.png", "test/N04.png", "test/O01.png", "test/O02.png", "test/P01.png", "test/P02.png", "test/P03.png", "test/P04.png", "test/P05.png", "test/P06.png", "test/P07.png", "test/P08.png", "test/R01.png", "test/R02.png", "test/R03.png", "test/S02.png", "test/S03.png", "test/S04.png", "test/S05.png", "test/S06.png", "test/S07.png", "test/S08.png", "test/T02.png", "test/U01.png", "test/U02.png", "test/U03.png", "test/U04.png", "test/V01.png", "test/W01.png", "test/W02.png", "test/X01.png", "test/X02.png", "test/X03.png", "test/X04.png", "test/Y01.png", "test/Y02.png", "test/Z01.png", "test/Z02.png", "test/cell10-gray_5x4.png", "test/cell10-gray_5x6.png" ];

inputFiles.forEach(function(fileName) {
  // console.log(fileName);
  cv.readImage(fileName, function(err, im) {
    im.detectObject(cascadeFile, {neighbors: 2, scale: 2}, function(err, objects) {
      /*if(err != undefined) {
         console.log(err);
         return;
      }*/
      console.log(objects);
      for(var k = 0; k < objects.length; k++) {
        var object = objects[k];
        im.rectangle([object.x, object.y], [object.width, object.height], color, 2);
      }
      im.save(fileName.replace(/.png/, "processed.png"));
    });
  });
});
