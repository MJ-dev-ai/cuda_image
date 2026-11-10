#include "core/device_buffer.hpp"

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : capacity_bytes_(other.capacity_bytes_), data_(other.data_) {
    other.capacity_bytes_ = 0;
    other.data_ = nullptr;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    capacity_bytes_ = other.capacity_bytes_;
    data_ = other.data_;
    other.capacity_bytes_ = 0;
    other.data_ = nullptr;
}