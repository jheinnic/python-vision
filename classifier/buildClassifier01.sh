#!/bin/sh

set +x

PWD=`pwd`
BG_COUNT=`wc -l ${PWD}/cellBgFiles.lst | awk -F\  '{print $1}'`
for letter in A B C D E F G H I K L M N O P Q R S T U V W X Y
do
   find ${PWD}/samples/${letter} -name '*.png' -print > ${PWD}/samples/${letter}Files.lst
   cat ${PWD}/samples/${letter}Files.lst | awk '{ print $1 " 1 0 0 73 75" }' > ${PWD}/samples/${letter}Sources.lst
done
cat ${PWD}/samples/[A-Z]Files.lst > ${PWD}/samples/allFiles.lst
for letter in A B C D E F G H I K L M N O P Q R S T U V W X Y
do
   grep -v "/${letter}/" ${PWD}/samples/allFiles.lst > ${PWD}/samples/no${letter}Fg.lst
   rm -f ${PWD}/samples/bgFiles${letter}.lst
   touch ${PWD}/samples/bgFiles${letter}.lst
   echo "rm -rf ${PWD}/generated/${letter}"
   echo "mkdir ${PWD}/generated/${letter}"
 
   # For each source image available, queue a run through the background file 
   # file list for both the current letter's  vec file construction, and also 
   # to contribute to every other letter's set of negative examples.
   ii=0
   numSources=`wc -l ${PWD}/samples/${letter}Sources.lst | awk -F\  '{print $1}'`
   cp ${PWD}/cellBgFiles.lst ${PWD}/samples/bgFiles${letter}.lst
   # while test $ii -lt $numSources;
   for srcFile in `cat ${PWD}/samples/${letter}Sources.lst`
   do
      echo "<$ii> -lt <$numSources>"
      # cat ${PWD}/cellBgFiles.lst >> ${PWD}/samples/bgFiles${letter}.lst
      opencv_createsamples -bg ${PWD}/samples/bgFiles${letter}.lst -img ${srcFile} -info ${PWD}/generated/${letter}/positiveSamples_${ii}.lst -maxxangle 0.1 -maxyangle 0.1 -maxzangle 0.1 -bgcolor 255  -maxidev 0 -maxscale 1 -w 73 -h 75 -num ${BG_COUNT}
      ii=`echo "$ii + 1" | bc -l`
   done

   echo "${BG_COUNT} * ${numSources}"
   numSamples=`echo "${BG_COUNT} * ${numSources}" | bc -l`
   echo "${numSamples}"
   echo opencv_createsamples -bg ${PWD}/samples/bgFiles${letter}Bg.lst -info ${PWD}/samples/${letter}Source.lst -maxxangle 0.1 -maxyangle 0.1 -maxzangle 0.1 -bgcolor 255  -maxidev 0 -maxscale 1 -w 73 -h 75 -vec ${PWD}/vectors/${letter}.vec -num ${numSamples}
   opencv_createsamples -info ${PWD}/samples/${letter}Source.lst -maxxangle 0.1 -maxyangle 0.1 -maxzangle 0.1 -bgcolor 255  -maxidev 0 -maxscale 1 -w 73 -h 75 -vec ${PWD}/vectors/${letter}.vec -num ${numSamples}
done

for letter in A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
do
   # TODO: Acquire generated sample list with brackgrounds from others.
   cat ${PWD}/blockFiles.lst ${PWD}/samples/no${letter}Fg.lst > ${PWD}/samples/negative${letter}.lst
   for otherLetter in A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
   do
      if test "x${otherLetter}" != "x${letter}"
      then
         cat ${PWD}/generated/${otherLetter}/positiveSamples_*.lst | awk -F\  '{ print $1}' >> ${PWD}/samples/negative${letter}.lst
      fi
      numNegative=`wc -l ${PWD}/samples/negative${letter}.lst | awk -F\  '{print $1}'`
      echo "opencv_traincascade -data ${PWD}/classifiers/${letter} -vec ${PWD}/vectors/${letter}.vec -bg ${PWD}/samples/negative${letter}.lst -numPos ${numSamples} -numNeg ${numNegative} -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -mode ALL -precalcValBufSize 1024 -precalcIdxBuxSize 1024 -w 73 -h 75"
      opencv_traincascade -data ${PWD}/classifiers/${letter} -vec ${PWD}/vectors/${letter}.vec -bg ${PWD}/samples/negative${letter}.lst -numPos ${numSamples} -numNeg ${numNegative} -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -mode ALL -precalcValBufSize 1024 -precalcIdxBuxSize 1024 -w 73 -h 75
   done
done

