package core.actors.units;

import utils.Vector2d;

import static core.TribesConfig.*;

public class Defender extends Unit
{
    public Defender(Vector2d pos, int kills, boolean isVeteran, int ownerID) {
        super(DEFENDER_ATTACK, DEFENDER_DEFENCE, DEFENDER_MOVEMENT, DEFENDER_MAX_HP, DEFENDER_RANGE, DEFENDER_COST, pos, kills, isVeteran, ownerID);
    }

    @Override
    public Defender copy() {
        Defender c = new Defender(getCurrentPosition(), getKills(), isVeteran(), getOwnerID());
        c.setCurrentHP(getCurrentHP());
        return c;
    }
}