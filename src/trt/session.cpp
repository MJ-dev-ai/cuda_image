#include "trt/session.hpp"
#include <QFile>
#include <QJsonDocument>
#include <QJsonParseError>

#include <stdexcept>

QJsonObject readSettings(const QString& config_path) {
    QFile file(config_path);
    if (!file.open(QIODevice::ReadOnly)) throw std::runtime_error("Cannot open settings file: " + config_path.toStdString());

    QJsonParseError parseError;
    const auto document  = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        throw std::runtime_error("Failed to parse settings file: " + config_path.toStdString() + ", error: " + parseError.errorString().toStdString());
    }

    return document.object();
}