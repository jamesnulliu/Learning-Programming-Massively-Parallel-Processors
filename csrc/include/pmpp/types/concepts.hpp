#pragma once

#include <iterator>
#include <ranges>

namespace pmpp
{

template <typename T>
concept is_readable_range = requires(T& t) {
    typename std::ranges::range_value_t<T>;
    { std::begin(t) } -> std::input_iterator;
    { std::end(t) } -> std::sentinel_for<std::ranges::iterator_t<T>>;
};

template <typename T>
concept is_range = is_readable_range<T>;

template <typename T>
concept is_writable_range = is_readable_range<T> && requires(T& t) {
    { std::begin(t) } -> std::output_iterator<std::ranges::range_value_t<T>>;
};

}  // namespace pmpp