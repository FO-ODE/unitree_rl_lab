#pragma once

#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include "isaaclab/envs/mdp/observations/observations.h"

namespace go2_deploy
{

class TerrainMapTopic
{
public:
    static TerrainMapTopic& instance()
    {
        static TerrainMapTopic topic;
        return topic;
    }

    ~TerrainMapTopic()
    {
        executor_.cancel();
        if (spin_thread_.joinable()) {
            spin_thread_.join();
        }
    }

    void start(const std::string& topic_name, std::size_t size)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (started_) {
            return;
        }

        data_.assign(size, 0.0f);
        node_ = std::make_shared<rclcpp::Node>("go2_policy_terrain_obs");
        sub_ = node_->create_subscription<std_msgs::msg::Float32MultiArray>(
            topic_name,
            rclcpp::SensorDataQoS(),
            [this, size](const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
                if (msg->data.size() != size) {
                    RCLCPP_WARN_THROTTLE(
                        node_->get_logger(),
                        *node_->get_clock(),
                        2000,
                        "Ignoring terrain map with size %zu, expected %zu",
                        msg->data.size(),
                        size
                    );
                    return;
                }
                std::lock_guard<std::mutex> callback_lock(mutex_);
                data_ = msg->data;
                received_ = true;
            }
        );

        executor_.add_node(node_);
        spin_thread_ = std::thread([this]() { executor_.spin(); });
        started_ = true;
    }

    std::vector<float> latest() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_;
    }

    bool received() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return received_;
    }

private:
    TerrainMapTopic() = default;

    mutable std::mutex mutex_;
    bool started_ = false;
    bool received_ = false;
    std::vector<float> data_;
    rclcpp::Node::SharedPtr node_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_;
    rclcpp::executors::SingleThreadedExecutor executor_;
    std::thread spin_thread_;
};

} // namespace go2_deploy

namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(terrain_map)
{
    const std::string topic = params["topic"] ? params["topic"].as<std::string>() : "/policy_terrain_obs";
    const std::size_t size = params["size"] ? params["size"].as<std::size_t>() : 187;
    auto& topic_reader = go2_deploy::TerrainMapTopic::instance();
    topic_reader.start(topic, size);
    return topic_reader.latest();
}

} // namespace mdp
} // namespace isaaclab
