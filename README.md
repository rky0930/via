# VIA Annotation tool
[VGG Image Annotator (VIA)](http://www.robots.ox.ac.uk/~vgg/software/via/)는 Image annotation tool입니다.  
Original repo는 이 [링크](https://gitlab.com/vgg/via/tags/via-1.0.4)와 같습니다.  

## New feature
Region을 선택하면 Select box가 생성되어 더 효율적인 Annotation 작업이 가능하게 합니다.  
VIA를 통해 생성된 annotation file을 tfrecord 형태로 생성하는 방법은 [링크]() 에서 확인하실 수 있습니다.  

(Screen Capture)
![](via_tesla.png)

## Installation
크롬 브라우져 (*별도 설치 필요없음)  
(모든 소스코드는 Javascript & HTML로 되어있습니다.)  
(All source code based on Javascript & HTML)

## Usage example
Object Detection 관련 데이터셋을 만들때 사용.  

## Development setup
HTML 수정 -> index.html  
Javascript -> via.js  
두 파일 통합  
```sh
sh pack_via.sh
```
최종 파일: via.html

## Author
Gyoung-yoon Ryoo – [rky0930@gmail.com]

## License
VIA is an open source project released under the 
BSD-2 clause license.

## Contributing

1. Fork it (<https://github.com/rky093/via/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

## Issue report 
불편한 사항이 있으면 issue report 부탁드립니다. 
