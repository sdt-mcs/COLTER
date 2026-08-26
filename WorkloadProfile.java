package com.perphproctor.common;

import com.perphproctor.common.WorkloadTypes.LraComponentType;

// Research-prototype subset extracted from the internal production system. The full profile
// carries additional per-component runtime metadata that cannot be open-sourced (enterprise
// compliance); the fields kept here are those the layer-aware prediction path (LAIP) needs.
public class WorkloadProfile {
    private final String componentId;
    private final String componentName;
    private boolean lra;
    private LraComponentType componentType;

    // Per-resource interference sensitivity (1-10), set per layer in WorkloadTypes.createLraProfile.
    private int cpuSensitivity    = 5;
    private int memorySensitivity = 5;
    private int llcSensitivity    = 5;
    private int mbwSensitivity    = 5;
    private int ioSensitivity     = 5;

    public WorkloadProfile(String componentId, String componentName) {
        this.componentId = componentId;
        this.componentName = componentName;
    }

    public String getComponentId()   { return componentId; }
    public String getComponentName() { return componentName; }

    public boolean isLra()          { return lra; }
    public void setLra(boolean lra) { this.lra = lra; }

    public LraComponentType getComponentType()          { return componentType; }
    public void setComponentType(LraComponentType type) { this.componentType = type; }

    public int getCpuSensitivity()          { return cpuSensitivity; }
    public void setCpuSensitivity(int v)    { this.cpuSensitivity = v; }
    public int getMemorySensitivity()       { return memorySensitivity; }
    public void setMemorySensitivity(int v) { this.memorySensitivity = v; }
    public int getLlcSensitivity()          { return llcSensitivity; }
    public void setLlcSensitivity(int v)    { this.llcSensitivity = v; }
    public int getMbwSensitivity()          { return mbwSensitivity; }
    public void setMbwSensitivity(int v)    { this.mbwSensitivity = v; }
    public int getIoSensitivity()           { return ioSensitivity; }
    public void setIoSensitivity(int v)     { this.ioSensitivity = v; }
}
