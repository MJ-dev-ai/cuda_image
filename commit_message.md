refactor: ImageBuffer에 Size 통합 및 자동 테스트 코드 제거

- Size 정의를 image_buffer.hpp로 이동하고 크기 비교 연산자 추가
- ImageBuffer에 Size 기반 생성자와 size() 접근자를 추가하고 내부 크기 저장 통합
- 기존 너비·높이 생성자와 접근자 호환성 유지
- 전처리에서 미할당 dst와 출력 크기를 확인하고 이동 대입으로 버퍼 준비
- 전처리와 GUI에서 Size 기반 생성 및 resize 호출 사용
- CUDA 회귀 테스트, pytest 래퍼 및 GUI 자동 테스트 코드 제거
- CMake, Makefile, 빌드 스크립트와 README의 테스트 항목 정리
