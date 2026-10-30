#pragma once
#include <cstdint>

namespace cv {

struct ColorRGBA;

struct ColorRGB {
    std::uint8_t r = 0;
    std::uint8_t g = 0;
    std::uint8_t b = 0;

    constexpr operator ColorRGBA() const noexcept;
};

struct ColorRGBA {
    std::uint8_t r = 0;
    std::uint8_t g = 0;
    std::uint8_t b = 0;
    std::uint8_t a = 255;

    constexpr operator ColorRGB() const noexcept { return {r, g, b}; }
};

constexpr ColorRGB::operator ColorRGBA() const noexcept { return {r, g, b, 255}; }

namespace colors {
inline constexpr ColorRGBA LIGHT_GRAY { 200, 200, 200, 255 }; // Light Gray
inline constexpr ColorRGBA GRAY { 130, 130, 130, 255 };      // Gray
inline constexpr ColorRGBA DARK_GRAY { 80, 80, 80, 255 };     // Dark Gray
inline constexpr ColorRGBA YELLOW { 253, 249, 0, 255 };      // Yellow
inline constexpr ColorRGBA GOLD { 255, 203, 0, 255 };        // Gold
inline constexpr ColorRGBA ORANGE { 255, 161, 0, 255 };      // Orange
inline constexpr ColorRGBA PINK { 255, 109, 194, 255 };      // Pink
inline constexpr ColorRGBA RED { 230, 41, 55, 255 };         // Red
inline constexpr ColorRGBA MAROON { 190, 33, 55, 255 };      // Maroon
inline constexpr ColorRGBA GREEN { 0, 228, 48, 255 };        // Green
inline constexpr ColorRGBA LIME { 0, 158, 47, 255 };         // Lime
inline constexpr ColorRGBA DARK_GREEN { 0, 117, 44, 255 };    // Dark Green
inline constexpr ColorRGBA SKY_BLUE { 102, 191, 255, 255 };   // Sky Blue
inline constexpr ColorRGBA BLUE { 0, 121, 241, 255 };        // Blue
inline constexpr ColorRGBA DARK_BLUE { 0, 82, 172, 255 };     // Dark Blue
inline constexpr ColorRGBA PURPLE { 200, 122, 255, 255 };    // Purple
inline constexpr ColorRGBA VIOLET { 135, 60, 190, 255 };     // Violet
inline constexpr ColorRGBA DARK_PURPLE { 112, 31, 126, 255 }; // Dark Purple
inline constexpr ColorRGBA BEIGE { 211, 176, 131, 255 };     // Beige
inline constexpr ColorRGBA BROWN { 127, 106, 79, 255 };      // Brown
inline constexpr ColorRGBA DARK_BROWN { 76, 63, 47, 255 };    // Dark Brown

inline constexpr ColorRGBA WHITE { 255, 255, 255, 255 }; // White
inline constexpr ColorRGBA BLACK { 0, 0, 0, 255 };       // Black
inline constexpr ColorRGBA BLANK { 0, 0, 0, 0 };         // Blank (Transparent)
inline constexpr ColorRGBA MAGENTA { 255, 0, 255, 255 }; // Magenta
} // namespace colors

} // namespace cv
