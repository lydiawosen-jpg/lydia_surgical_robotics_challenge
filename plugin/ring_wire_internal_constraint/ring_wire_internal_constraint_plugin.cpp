#include <afFramework.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

using namespace ambf;
using namespace chai3d;

namespace {

constexpr double kEpsilon = 1e-8;
constexpr double kWireRadius = 0.0064;
constexpr double kLongitudinalLimit = 1.0;
constexpr double kAngularLimit = 0.7853981633974483;

double distance(const cVector3d& first, const cVector3d& second)
{
    return (first - second).length();
}

cVector3d getMeshCentroid(cMultiMesh* mesh)
{
    cVector3d centroid(0.0, 0.0, 0.0);
    if (mesh == nullptr || mesh->getNumVertices() == 0) {
        return centroid;
    }
    for (unsigned int i = 0; i < mesh->getNumVertices(); ++i) {
        centroid += mesh->getVertexPos(i);
    }
    return centroid * (1.0 / static_cast<double>(mesh->getNumVertices()));
}

std::vector<cVector3d> extractCenterline(cMultiMesh* mesh)
{
    if (mesh == nullptr || mesh->m_vtxIdxMap.empty()) {
        return {};
    }

    std::vector<std::vector<cVector3d>> candidates;
    candidates.reserve(mesh->m_vtxIdxMap.size());
    for (const auto& [originalIndex, importedIndices] : mesh->m_vtxIdxMap) {
        (void)originalIndex;
        std::vector<cVector3d> vertexCandidates;
        for (const int importedIndex : importedIndices) {
            if (importedIndex < 0) {
                return {};
            }
            for (int meshIndex = 0; meshIndex < mesh->getNumMeshes(); ++meshIndex) {
                cMesh* childMesh = mesh->getMesh(meshIndex);
                if (childMesh == nullptr ||
                    static_cast<unsigned int>(importedIndex) >= childMesh->getNumVertices()) {
                    continue;
                }
                const cVector3d candidate = childMesh->m_vertices->getLocalPos(importedIndex);
                const bool duplicate = std::any_of(
                    vertexCandidates.begin(),
                    vertexCandidates.end(),
                    [&candidate](const cVector3d& existing) {
                        return distance(existing, candidate) <= kEpsilon;
                    });
                if (!duplicate) {
                    vertexCandidates.push_back(candidate);
                }
            }
        }
        if (vertexCandidates.empty()) {
            return {};
        }
        candidates.push_back(std::move(vertexCandidates));
    }

    std::vector<std::vector<double>> costs(candidates.size());
    std::vector<std::vector<size_t>> predecessors(candidates.size());
    costs.front().assign(candidates.front().size(), 0.0);
    for (size_t i = 1; i < candidates.size(); ++i) {
        costs[i].assign(candidates[i].size(), std::numeric_limits<double>::infinity());
        predecessors[i].resize(candidates[i].size());
        for (size_t current = 0; current < candidates[i].size(); ++current) {
            for (size_t previous = 0; previous < candidates[i - 1].size(); ++previous) {
                const double cost = costs[i - 1][previous] +
                                    distance(candidates[i - 1][previous], candidates[i][current]);
                if (cost < costs[i][current]) {
                    costs[i][current] = cost;
                    predecessors[i][current] = previous;
                }
            }
        }
    }

    size_t selected = std::distance(
        costs.back().begin(),
        std::min_element(costs.back().begin(), costs.back().end()));
    std::vector<cVector3d> vertices(candidates.size());
    for (size_t i = candidates.size(); i-- > 0;) {
        vertices[i] = candidates[i][selected];
        if (i > 0) {
            selected = predecessors[i][selected];
        }
    }

    std::vector<double> consecutiveDistances;
    consecutiveDistances.reserve(vertices.size() - 1);
    for (size_t i = 1; i < vertices.size(); ++i) {
        consecutiveDistances.push_back(distance(vertices[i - 1], vertices[i]));
    }
    if (consecutiveDistances.empty()) {
        return {};
    }
    std::vector<double> sortedDistances = consecutiveDistances;
    std::sort(sortedDistances.begin(), sortedDistances.end());
    const double typicalEdgeLength = sortedDistances[sortedDistances.size() / 2];
    if (typicalEdgeLength <= kEpsilon) {
        return {};
    }

    const double ringBreakThreshold = typicalEdgeLength * 4.0;
    size_t verticesPerRing = 0;
    for (size_t i = 0; i < consecutiveDistances.size(); ++i) {
        if (consecutiveDistances[i] > ringBreakThreshold) {
            verticesPerRing = i + 1;
            break;
        }
    }
    if (verticesPerRing < 3 || vertices.size() / verticesPerRing < 2) {
        return {};
    }

    std::vector<cVector3d> centerline;
    centerline.reserve(vertices.size() / verticesPerRing);
    for (size_t first = 0; first + verticesPerRing <= vertices.size(); first += verticesPerRing) {
        cVector3d center(0.0, 0.0, 0.0);
        for (size_t i = 0; i < verticesPerRing; ++i) {
            center += vertices[first + i];
        }
        centerline.push_back(center * (1.0 / static_cast<double>(verticesPerRing)));
    }
    return centerline;
}

struct ClosestCenterlinePoint
{
    cVector3d point;
    cVector3d tangent;
};

}  // namespace

class afRingWireInternalConstraintPlugin : public afModelPlugin
{
public:
    int init(const afModelPtr a_afModel, const afModelAttribsPtr a_modelAttribs) override
    {
        m_modelPtr = a_afModel;
        a_attribs = a_modelAttribs;
        if (m_modelPtr == nullptr) {
            std::cerr << "[ring_wire_internal_constraint] ERROR: model pointer is null" << std::endl;
            return afInitStatus::ERROR;
        }

        ringBody_ = m_modelPtr->getRigidBody("ring_visual", true);
        wireBody_ = m_modelPtr->getRigidBody("wire_visual", true);
        anchorBody_ = m_modelPtr->getRigidBody("wire_anchor", true);
        if (ringBody_ == nullptr || wireBody_ == nullptr || anchorBody_ == nullptr ||
            ringBody_->m_bulletRigidBody == nullptr || wireBody_->m_bulletRigidBody == nullptr ||
            anchorBody_->m_bulletRigidBody == nullptr) {
            std::cerr << "[ring_wire_internal_constraint] ERROR: ring_visual, wire_visual, or wire_anchor is unavailable"
                      << std::endl;
            return afInitStatus::ERROR;
        }

        cMultiMesh* ringMesh = ringBody_->getVisualObject();
        if (ringMesh == nullptr || ringMesh->getNumVertices() == 0) {
            std::cerr << "[ring_wire_internal_constraint] ERROR: ring_visual has no visual mesh" << std::endl;
            return afInitStatus::ERROR;
        }
        ringCenterLocal_ = getMeshCentroid(ringMesh);
        double innerRadius = std::numeric_limits<double>::infinity();
        for (unsigned int i = 0; i < ringMesh->getNumVertices(); ++i) {
            const cVector3d delta = ringMesh->getVertexPos(i) - ringCenterLocal_;
            innerRadius = std::min(innerRadius, std::hypot(delta.x(), delta.y()));
        }
        lateralClearance_ = innerRadius - kWireRadius;
        if (!std::isfinite(lateralClearance_) || lateralClearance_ <= kEpsilon) {
            std::cerr << "[ring_wire_internal_constraint] ERROR: invalid ring clearance" << std::endl;
            return afInitStatus::ERROR;
        }

        centerlineLocal_ = extractCenterline(wireBody_->getVisualObject());
        if (centerlineLocal_.size() < 2) {
            std::cerr << "[ring_wire_internal_constraint] ERROR: could not extract wire centerline" << std::endl;
            return afInitStatus::ERROR;
        }

        const ClosestCenterlinePoint closest = findClosestPoint();
        const btTransform anchorTransform = makeAnchorTransform(closest);
        setAnchorTransform(anchorTransform);

        const btTransform ringTransform = ringBody_->m_bulletRigidBody->getCenterOfMassTransform();
        const btTransform frameInAnchor = anchorTransform.inverse() * anchorTransform;
        const btTransform frameInRing = ringTransform.inverse() * anchorTransform;
        constraint_ = new btGeneric6DofConstraint(
            *anchorBody_->m_bulletRigidBody,
            *ringBody_->m_bulletRigidBody,
            frameInAnchor,
            frameInRing,
            true);
        constraint_->setLinearLowerLimit(btVector3(
            -lateralClearance_,
            -lateralClearance_,
            -kLongitudinalLimit));
        constraint_->setLinearUpperLimit(btVector3(
            lateralClearance_,
            lateralClearance_,
            kLongitudinalLimit));
        constraint_->setAngularLowerLimit(btVector3(-kAngularLimit, -kAngularLimit, -kAngularLimit));
        constraint_->setAngularUpperLimit(btVector3(kAngularLimit, kAngularLimit, kAngularLimit));
        anchorBody_->m_afWorld->m_bulletWorld->addConstraint(constraint_, true);

        std::cerr << "[ring_wire_internal_constraint] initialized 6DOF anchor constraint with lateral clearance="
                  << lateralClearance_ << ", longitudinal limit=" << kLongitudinalLimit
                  << ", angular limit=" << kAngularLimit << std::endl;
        return afInitStatus::SUCCESS;
    }

    void physicsUpdate(double) override
    {
        if (constraint_ == nullptr || ringBody_ == nullptr || wireBody_ == nullptr || anchorBody_ == nullptr) {
            return;
        }
        setAnchorTransform(makeAnchorTransform(findClosestPoint()));
    }

    bool close() override
    {
        if (constraint_ != nullptr && anchorBody_ != nullptr && anchorBody_->m_afWorld != nullptr &&
            anchorBody_->m_afWorld->m_bulletWorld != nullptr) {
            anchorBody_->m_afWorld->m_bulletWorld->removeConstraint(constraint_);
        }
        delete constraint_;
        constraint_ = nullptr;
        ringBody_ = nullptr;
        wireBody_ = nullptr;
        anchorBody_ = nullptr;
        centerlineLocal_.clear();
        return true;
    }

private:
    ClosestCenterlinePoint findClosestPoint() const
    {
        const btTransform ringTransform = ringBody_->m_bulletRigidBody->getCenterOfMassTransform();
        const cVector3d ringCenter(
            ringTransform.getOrigin().x(),
            ringTransform.getOrigin().y(),
            ringTransform.getOrigin().z());
        const btTransform wireTransform = wireBody_->m_bulletRigidBody->getCenterOfMassTransform();
        double closestDistanceSquared = std::numeric_limits<double>::infinity();
        ClosestCenterlinePoint closest;
        for (size_t i = 1; i < centerlineLocal_.size(); ++i) {
            const btVector3 firstBt = wireTransform * btVector3(
                centerlineLocal_[i - 1].x(),
                centerlineLocal_[i - 1].y(),
                centerlineLocal_[i - 1].z());
            const btVector3 secondBt = wireTransform * btVector3(
                centerlineLocal_[i].x(),
                centerlineLocal_[i].y(),
                centerlineLocal_[i].z());
            const cVector3d first(firstBt.x(), firstBt.y(), firstBt.z());
            const cVector3d second(secondBt.x(), secondBt.y(), secondBt.z());
            const cVector3d segment = second - first;
            const double segmentLengthSquared = segment.lengthsq();
            if (segmentLengthSquared <= kEpsilon) {
                continue;
            }
            const double fraction = std::clamp(
                (ringCenter - first).dot(segment) / segmentLengthSquared,
                0.0,
                1.0);
            const cVector3d point = first + segment * fraction;
            const double distanceSquared = (ringCenter - point).lengthsq();
            if (distanceSquared < closestDistanceSquared) {
                closestDistanceSquared = distanceSquared;
                closest = {point, segment * (1.0 / std::sqrt(segmentLengthSquared))};
            }
        }
        return closest;
    }

    btTransform makeAnchorTransform(const ClosestCenterlinePoint& closest) const
    {
        btVector3 tangent(closest.tangent.x(), closest.tangent.y(), closest.tangent.z());
        btVector3 reference(0.0, 0.0, 1.0);
        if (std::abs(tangent.dot(reference)) > 0.9) {
            reference = btVector3(0.0, 1.0, 0.0);
        }
        const btVector3 axisX = tangent.cross(reference).normalized();
        const btVector3 axisY = tangent.cross(axisX).normalized();
        btMatrix3x3 basis(
            axisX.x(), axisY.x(), tangent.x(),
            axisX.y(), axisY.y(), tangent.y(),
            axisX.z(), axisY.z(), tangent.z());
        return btTransform(
            basis,
            btVector3(closest.point.x(), closest.point.y(), closest.point.z()));
    }

    void setAnchorTransform(const btTransform& transform) const
    {
        btRigidBody* anchor = anchorBody_->m_bulletRigidBody;
        anchor->setCenterOfMassTransform(transform);
        anchor->setWorldTransform(transform);
        if (anchor->getMotionState() != nullptr) {
            anchor->getMotionState()->setWorldTransform(transform);
        }
    }

    afRigidBodyPtr ringBody_ = nullptr;
    afRigidBodyPtr wireBody_ = nullptr;
    afRigidBodyPtr anchorBody_ = nullptr;
    btGeneric6DofConstraint* constraint_ = nullptr;
    cVector3d ringCenterLocal_ = cVector3d(0.0, 0.0, 0.0);
    std::vector<cVector3d> centerlineLocal_;
    double lateralClearance_ = 0.0;
};

AF_REGISTER_MODEL_PLUGIN(afRingWireInternalConstraintPlugin)